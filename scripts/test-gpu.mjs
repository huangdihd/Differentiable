import { spawn } from 'node:child_process';
import { access, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createServer } from 'vite';

// No Playwright dependency and no fallback to software rendering or CPU execution.
const candidates = [
  process.env.CHROME_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].filter(Boolean);
let chrome, profile, server, child, timer;
let stderr = '';
try {
  for (const path of candidates) {
    try {
      await access(path);
      chrome = path;
      break;
    } catch {
      /* next installed browser */
    }
  }
  if (!chrome)
    throw new Error('Chrome not found: set CHROME_PATH to an installed Chrome executable');
  try {
    profile = await mkdtemp(join(tmpdir(), 'differentiable-gpu-'));
  } catch (error) {
    throw new Error(
      `Cannot create Chrome temporary profile in os.tmpdir() (${tmpdir()}); grant temporary-directory access. ${error}`,
    );
  }
  let resolve, reject;
  const completed = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  server = await createServer({
    configFile: false,
    server: { host: '127.0.0.1', port: 0 },
    plugins: [
      {
        name: 'gpu-test-report',
        configureServer(vite) {
          vite.middlewares.use('/__gpu_report', (req, res) => {
            if (req.method !== 'POST') {
              res.statusCode = 405;
              res.end();
              return;
            }
            let body = '';
            req.on('data', (part) => {
              body += part;
              if (body.length > 1000000) req.destroy();
            });
            req.on('end', () => {
              try {
                const report = JSON.parse(body);
                console.log(JSON.stringify(report));
                res.end('ok');
                if (report.type === 'done') resolve(report);
                if (report.type === 'failure') reject(new Error(report.error));
              } catch (error) {
                res.statusCode = 400;
                res.end();
                reject(error);
              }
            });
          });
        },
      },
    ],
  });
  await server.listen();
  const url = `${server.resolvedUrls.local[0]}gpu-tests.html`;
  child = spawn(
    chrome,
    [
      '--headless=new',
      '--no-first-run',
      ...(process.env.GPU_DISABLE_CRASH_REPORTER === '1' ? ['--disable-crash-reporter', '--disable-breakpad'] : []),
      '--no-default-browser-check',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
      '--enable-unsafe-webgpu',
      `--user-data-dir=${profile}`,
      url,
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  );
  child.stderr.on('data', (data) => {
    stderr = (stderr + data).slice(-12000);
  });
  child.once('error', reject);
  child.once('exit', (code, signal) =>
    reject(new Error(`Chrome exited before completion (${code}, ${signal})`)),
  );
  timer = setTimeout(
    () => reject(new Error('GPU browser tests timed out')),
    Number(process.env.GPU_TEST_TIMEOUT_MS ?? 1800000),
  );
  await completed;
} catch (error) {
  console.error(`GPU tests FAILED: ${error.stack ?? error}`);
  if (stderr) console.error(stderr);
  process.exitCode = 1;
} finally {
  clearTimeout(timer);
  if (child && child.exitCode === null && child.signalCode === null) {
    const exited = new Promise((resolve) => child.once('exit', resolve));
    child.kill('SIGTERM');
    const kill = setTimeout(() => child.kill('SIGKILL'), 3000);
    await exited;
    clearTimeout(kill);
  }
  await server?.close();
  if (profile) await rm(profile, { recursive: true, force: true, maxRetries: 3 });
}

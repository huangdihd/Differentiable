// Obfuscates the built JavaScript in dist/assets for the public demo build.
// Only cheap transforms: the tensor engine runs tight loops, and control-flow flattening,
// dead-code injection or base64 string decoding would slow training (base64 doubled XOR's
// training time; the plain string array does not measurably).
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import JavaScriptObfuscator from 'javascript-obfuscator';

const dir = new URL('../dist/assets/', import.meta.url).pathname;

const options = {
  compact: true,
  target: 'browser',
  identifierNamesGenerator: 'hexadecimal',
  renameGlobals: false, // ES module exports and the worker's globals must keep their names
  stringArray: true,
  stringArrayEncoding: [],
  stringArrayThreshold: 0.75,
  stringArrayRotate: true,
  stringArrayShuffle: true,
  splitStrings: false,
  controlFlowFlattening: false,
  deadCodeInjection: false,
  selfDefending: false,
  debugProtection: false,
  transformObjectKeys: false, // locale keys and graph JSON are looked up by name
  unicodeEscapeSequence: false,
  sourceMap: false,
};

for (const name of await readdir(dir)) {
  if (!name.endsWith('.js')) continue;
  const file = join(dir, name);
  const source = await readFile(file, 'utf8');
  const result = JavaScriptObfuscator.obfuscate(source, options).getObfuscatedCode();
  await writeFile(file, result);
  console.log(`obfuscated ${name}: ${(source.length / 1024).toFixed(0)} kB → ${(result.length / 1024).toFixed(0)} kB`);
}

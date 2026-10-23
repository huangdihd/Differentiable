-- 用法：npx wrangler d1 execute differentiable-analytics --remote --file analytics/queries.sql
-- 或者把单条查询用 --command "..." 执行。

-- 1. 每关漏斗：进入人数、通关人数、通关率
SELECT level, chapter,
  COUNT(DISTINCT CASE WHEN event = 'enter' THEN visitor END) AS entered,
  COUNT(DISTINCT CASE WHEN event = 'complete' THEN visitor END) AS completed,
  ROUND(100.0 * COUNT(DISTINCT CASE WHEN event = 'complete' THEN visitor END)
    / MAX(COUNT(DISTINCT CASE WHEN event = 'enter' THEN visitor END), 1), 1) AS completion_pct
FROM events GROUP BY level, chapter ORDER BY chapter, level;

-- 2. 每关通关用时和尝试次数（中位数近似用平均值；单次访问内）
SELECT level, COUNT(*) AS completions,
  ROUND(AVG(ms) / 1000.0, 1) AS avg_seconds,
  ROUND(AVG(attempts), 2) AS avg_attempts
FROM events WHERE event = 'complete' GROUP BY level ORDER BY level;

-- 3. 每关放弃：离开时还没通关、且从未通关过该关的玩家数
SELECT level, COUNT(DISTINCT visitor) AS abandoned_visitors,
  ROUND(AVG(ms) / 1000.0, 1) AS avg_seconds_before_leaving
FROM events e WHERE event = 'leave' AND passed = 0
  AND NOT EXISTS (SELECT 1 FROM events c WHERE c.event = 'complete'
    AND c.visitor = e.visitor AND c.level = e.level)
GROUP BY level ORDER BY level;

-- 4. 每章到达人数（最远到达哪一章）
SELECT chapter, COUNT(DISTINCT visitor) AS visitors
FROM events WHERE event = 'enter' GROUP BY chapter ORDER BY chapter;

-- 5. 每日活跃访客
SELECT date(received_at / 1000, 'unixepoch') AS day, COUNT(DISTINCT visitor) AS visitors
FROM events GROUP BY day ORDER BY day;

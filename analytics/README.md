# 关卡统计

- 前端：`src/analytics.ts`，进入关卡、每次判题、通关、离开（切走标签页 / 关闭页面 / 退出关卡）时用 `sendBeacon` 发到 `/api/event`。只发送随机的访客 id（存在 localStorage）、会话 id、关卡号、用时和尝试次数，不收集任何个人信息。localhost 和自动化浏览器不发送。任何错误都会被吞掉，不影响游戏。
- 后端：`functions/api/event.ts`（Cloudflare Pages Function），校验后写入 D1 的 `events` 表；没有绑定 D1 时直接丢弃。限制请求体 1 KB、每个访客每分钟 60 条（单实例内的粗略限制），过滤常见爬虫 UA。
- 表结构：`migrations/0001_events.sql`。统计查询：`analytics/queries.sql`。没有公开的读取接口，只能用 wrangler 或 Cloudflare 控制台查询。

## 配置

```sh
npx wrangler d1 create differentiable-analytics
npx wrangler d1 execute differentiable-analytics --remote --file migrations/0001_events.sql
```

然后在 Cloudflare 控制台 → Workers & Pages → differentiable → 设置 → 绑定，添加 D1 数据库绑定，变量名 `DB`，选 `differentiable-analytics`（生产环境）。

# Differentiable

[English](README.md) · **简体中文**

一个跑在浏览器里的神经网络搭建游戏。从 `a + b` 开始，最后搭出一个能训练、能写字的 GPT——
你亲手连的每一个元件都会进元件库，下一关继续用。

[**在线试玩**](https://differentiable.pages.dev) · [宣传视频](https://www.bilibili.com/video/BV13BJt6SEVx) ·
[引擎架构](docs/architecture.md) · [关卡设计](docs/levels.md)

[![Deploy demo](https://github.com/huangdihd/Differentiable/actions/workflows/deploy-demo.yml/badge.svg)](https://github.com/huangdihd/Differentiable/actions/workflows/deploy-demo.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

## 为什么做这个

其他自动求导教程都直接给你 `loss.backward()`。这里反过来：在自动求导工具解锁之前，反向传播是
你要亲手连出来的电路——链式法则不是被你调用的，是被你搭出来的。softmax、LayerNorm、注意力、
Transformer block 全都一样，每一关都先解出来，再收进自己的库。

游戏底下是一个真的引擎：带类型的张量、版本化的 JSON 图、一阶反向自动求导、CPU 参考后端和
WebGPU 后端。不是那种为了教学把答案写死的玩具。

## 你会搭出什么

| 章节 | 主题 | 最后搭成 |
| --- | --- | --- |
| 0 | 标量 | `sigmoid`、`tanh` |
| 1 | 求导 | 亲手连的 VJP，然后才解锁自动求导 |
| 2 | 张量 | 广播、matmul、softmax、LayerNorm |
| 3 | 学习 | MSE、交叉熵、SGD、动量、Adam、MLP——异或 |
| 4 | 序列 | embedding、bigram、双字符模型 |
| 5 | 时序 | 固定窗口的失败、RNN、末状态瓶颈 |
| 6 | 注意力 | 注意力、因果掩码、位置编码 |
| 7 | GPT | 多头注意力、前馈、残差块，一个能训练的 GPT |

共 62 关。约 3500 个测试覆盖引擎、关卡目录和训练路径。

## 快速开始

需要 Node 22.12+、24+ 或 26+。

```sh
npm install
npm run dev        # http://localhost:5173
```

```sh
npm run typecheck
npm test           # 非 watch；光 GPT 校准测试就要约 67 秒
npm run build
npm run format:check
```

完整环境说明、demo 部署流程和已知限制见 [docs/development.md](docs/development.md)。

## 目录结构

```
src/engine/    张量、图编译、自动求导、训练、后端（CPU + WebGPU）
src/levels/    关卡目录、参考电路、带种子的检查
src/editor/    图文档状态、撤销/重做、存档迁移
src/ui/        React 编辑器外壳、关卡面板、图表、示意图
src/worker/    主线程之外的训练、生成与评测
src/locales/   en 与 zh-CN 的扁平键词典
demos/         早期原型，留作参考，不是应用源码
```

## 文档

- [引擎架构](docs/architecture.md)——后端契约、图格式、自动求导、元件展开、WebGPU 选择与限制
- [关卡设计](docs/levels.md)——关卡路线、稳定 ID、存档兼容、各章校准数据
- [开发](docs/development.md)——环境、脚本、demo 部署、当前限制
- [编辑器内部](src/ui/README.md) · [Worker 协议](src/worker/README.md) ·
  [音频](src/audio/README.md) · [统计分析](analytics/README.md)

## 参与贡献

欢迎提 issue 和 PR。有两条规矩比其他的都重要：

- **关卡 ID 就是存档格式。** 显示编号可以改，关卡的 `id` 字段不行。改一个名字就会静默废掉
  所有人的存档。
- **行为变更要先有一个会失败的测试。** 形状推断、梯度边界情况、训练确定性都以测试为准；
  PR 之前 `npm run typecheck` 和 `npm test` 必须通过。

提交前跑一次 `npm run format`。格式是 Prettier，100 列、单引号。

## 许可证

Differentiable 是自由软件：你可以依照自由软件基金会发布的 GNU 通用公共许可证（第 3 版或你选择
的任何更新版本）的条款重新发布和/或修改它。

本程序是基于「希望它有用」而发布的，**不提供任何担保**，甚至不提供适销性或特定用途适用性的
默示担保。完整条款见 [`LICENSE`](LICENSE)。

每个源文件都带 `SPDX-License-Identifier: GPL-3.0-or-later`。第三方依赖与字体（均为与 GPLv3
兼容的许可证）列在 [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md)。

线上的 demo 由 `npm run build:demo` 混淆后发布。混淆不会让它变得不自由：GPLv3 第 6 条要求
任何收到该产物的人都能取得对应源码，而本仓库就是那份源码。只要 demo 还在公开访问，仓库就
必须保持公开。

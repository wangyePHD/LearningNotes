# DSH + dsh-TUI 服务器安装手册（支持 1017 版模型）

> **用途**：交给 Claude Code（或任何执行型 Agent）在 Linux 服务器上按步骤安装。
> **网络前提**：服务器可访问 GitHub 与 npm；**笔记本无需与服务器直连**，全程在服务器上完成。
> **目标组合**：`@deepseek-ai/dsh@0.2.1-rc.2` + `dsh-tui 0.15.0（commit e709863 源码构建）`
> **验收标准**：TUI 的 `/model` 列表中出现 **`DeepSeek-V4.1-Flash 1017`**，选中后能正常流式对话。
> **文档日期**：2026-10-18

---

## 0. 先读：为什么必须锁这两个版本

| 组件 | 必须使用 | 原因 |
|---|---|---|
| `@deepseek-ai/dsh` | **0.2.1-rc.2** | 1017 适配基线，2026-10-17 发布。npm 的 `latest` 仍是 `0.2.0-rc.2`，**必须显式写版本号** |
| `dsh-tui` | **源码 `e709863` 构建** | npm 上的 0.15.0 发布于 2026-10-13，**不含 1017 标记**，peer 也只声明到 `0.2.0-rc.2`；1017 版本尚未发 npm |
| Node.js | `^22.19 \|\| >=24` | `dsh-tui` 的 `engines.node` 硬性要求 |
| pnpm | **11.21.0** | 仓库 `packageManager` 字段锁定；pnpm 9 会导致依赖提升异常 |

**关于 1017**：它是 `DeepSeek-V4.1-Flash` 的发布标记，**API model id 仍然是 `deepseek-flash`**；标记只影响模型列表里的显示名，不改变请求路由。所以"支持 1017"= 用 `deepseek-flash` 正常调用 + 界面能正确标出 1017。

**关于 dsh-harness**：不需要单独安装。Harness 内核（`@deepseek-ai/dsh-base` 及几十个 `@deepseek-ai/dsh-*` 包）是 `@deepseek-ai/dsh` 的依赖，随 CLI 一起装上。`deepseek-harness` 是项目名，不是要另装的包。

**为什么在服务器上构建**：1017 版 TUI 只存在于 GitHub 源码，没发 npm；服务器既然能上网，直接 `git clone` + 构建即可，不需要从笔记本传任何文件。

---

## 1. 前置条件

- Ubuntu 22.04 / 24.04（Debian 12 同理），x86_64 或 arm64
- 2 核 4G 起步，**磁盘 ≥ 8 GB**（源码构建 + 依赖峰值约 4–6 GB；装完可清理到 ~2 GB）
- 普通用户 + sudo（不要用 root 跑 TUI）
- 出网：`github.com`、`registry.npmjs.org`、`api.deepseek.com`
- 一个有效的 DeepSeek API Key（`sk-` 开头）
- 不需要笔记本与服务器互通，不需要 scp

---

## 2. 服务器基础环境

```bash
sudo apt-get update
sudo apt-get install -y curl git ca-certificates tmux build-essential
```

安装 Node.js 24（满足 `>=24`）：

```bash
curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
sudo apt-get install -y nodejs
node -v    # 期望 v24.x
```

安装 pnpm（与仓库 `packageManager: pnpm@11.21.0` 对齐）：

```bash
sudo npm install -g pnpm@11.21.0
pnpm -v    # 期望 11.21.0
```

> 国内服务器若下载慢，可临时换镜像：
> `npm config set registry https://registry.npmmirror.com`
> 换镜像后要确认 `@deepseek-ai/*` 也能命中，装完核对版本号。

---

## 3. 安装 dsh CLI（0.2.1-rc.2）

```bash
sudo npm install -g @deepseek-ai/dsh@0.2.1-rc.2
dsh --version
```

**期望输出**：`0.2.1-rc.2`

如果输出是 `0.2.0-rc.2`，说明装到了 `latest`，重跑上面的命令并确认版本号写全。

---

## 4. 在服务器上构建并安装 dsh-tui（1017 构建）

### 4.1 克隆源码（必须带子模块）

```bash
cd ~
git clone --recurse-submodules https://github.com/ccch1mneyyy/dsh-TUI.git
cd dsh-TUI
git checkout e709863019d56897e64495699c3a945cce194d00
```

确认子模块已就位（**为空则后面必然编译失败**）：

```bash
ls vendor/dsh-std/packages/core/package.json
```

### 4.2 安装依赖并构建

```bash
pnpm install --frozen-lockfile
pnpm build
```

- 耗时约 10–20 分钟，下载依赖 1–2 GB，属正常
- `pnpm build` = 编译 + 产物校验
- 若仅在最后的 verify 阶段失败（环境相关），可退回只编译：`pnpm compile`

**构建报错时**：先看下面第 10 节，不要直接改源码。

### 4.3 打包并确认包含 1017

```bash
npm pack --ignore-scripts
ls -lh deepseek-harness-tui-dsh-tui-0.15.0.tgz

tar -xzOf deepseek-harness-tui-dsh-tui-0.15.0.tgz \
  package/lib/types/dsh-adapter/channel/model-actions.js | grep -c "1017"
```

**期望**：最后一行输出 `1`（表示 1017 标记代码在包里）。输出 `0` 就**停止**，检查是否 checkout 到了 `e709863`。

> `--ignore-scripts` 必须加：仓库的 `prepare` 会二次编译且自带 git 守卫，打包阶段不需要再跑。

### 4.4 放到固定位置

```bash
sudo mkdir -p /opt/dsh
sudo chown "$USER":"$USER" /opt/dsh
mv deepseek-harness-tui-dsh-tui-0.15.0.tgz /opt/dsh/dsh-tui-0.15.0-1017.tgz
ls -lh /opt/dsh/
```

> ⚠️ `/opt/dsh/` 下的 tgz 不要删。profile 的 `package.json` 会记录这个 `file:` 路径，删了以后重装/加插件会失败。

### 4.5 安装全局启动器

```bash
sudo npm install -g --ignore-scripts /opt/dsh/dsh-tui-0.15.0-1017.tgz
dsh-tui version
```

**期望输出**（profile 尚未安装是正常的）：

```text
@deepseek-harness-tui/dsh-tui 0.15.0 (launcher)
profile: （未安装）  /home/<user>/.dsh/profiles/dsh-tui/node_modules/...
```

### 4.6 初始化 profile（**关键步骤，顺序不能错**）

```bash
dsh plugin --profile dsh-tui add file:/opt/dsh/dsh-tui-0.15.0-1017.tgz
```

这一步会：
1. 创建 `~/.dsh/profiles/dsh-tui/`
2. 装上 `@deepseek-ai/dsh-base` 等基础依赖
3. 通过 pnpm 安装你构建的 tgz，并把它注册为 bundle 层

耗时约 2–6 分钟，下载量几百 MB。

**遇到 `ERR_PNPM_IGNORED_BUILDS`**（pnpm ≥11 拦截安装脚本）：

```bash
cd ~/.dsh/profiles/dsh-tui
printf "\nallowBuilds:\n  '@google/genai': false\n  protobufjs: false\n" >> pnpm-workspace.yaml
dsh plugin --profile dsh-tui add file:/opt/dsh/dsh-tui-0.15.0-1017.tgz
```

**遇到 `ERR_PNPM_PEER_DEP_ISSUES` 且提示 `@deepseek-ai/dsh-app-boot` 版本不符**：
说明装的不是 1017 构建，回到 4.3 检查 tgz 里是否含 1017。

> ❗ **不要在 4.6 之前先运行 `dsh-tui`**。启动器首次运行会自举 npm 上的 0.15.0（10-13 版，无 1017），把 profile 装错。已经装错的话执行：
> `dsh plugin --profile dsh-tui add file:/opt/dsh/dsh-tui-0.15.0-1017.tgz` 覆盖即可。

### 4.7 验证装的是 1017 构建

```bash
dsh-tui version
grep -rl "1017" ~/.dsh/profiles/dsh-tui/node_modules/@deepseek-harness-tui/dsh-tui/lib | head -1
grep '"@deepseek-harness-tui/dsh-tui"' ~/.dsh/profiles/dsh-tui/package.json
```

**期望**：
- `dsh-tui version` 的 profile 行显示 `0.15.0`
- `grep` 命中一个 js 文件（如 `.../dsh-adapter/channel/model-actions.js`）
- `package.json` 里的依赖是 `file:/opt/dsh/dsh-tui-0.15.0-1017.tgz`

---

## 5. 配置 API Key

### 方式 A：环境变量（推荐）

```bash
mkdir -p ~/.dsh
install -m 600 /dev/null ~/.dsh/env
echo 'export DEEPSEEK_API_KEY="sk-你的key"' >> ~/.dsh/env
echo '[ -f "$HOME/.dsh/env" ] && . "$HOME/.dsh/env"' >> ~/.bashrc
. ~/.dsh/env
```

> 用 zsh 的话，把写入 `~/.bashrc` 的那行改写到 `~/.zshrc`。

先用 API 自检 key 有效：

```bash
curl -s https://api.deepseek.com/user/balance \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY"
```

返回 `{"is_available":true,...}` 才继续。

### 方式 B：账号登录（可选）

TUI 内 `/auth login deepseek-account`。服务器无浏览器，需要 SSH 端口转发配合；除非你明确要用账号额度，否则**用方式 A**。

---

## 6. 首次启动

TUI 必须有真实 TTY，**不能在管道/重定向里启动**，服务器上用 tmux：

```bash
tmux new -s dsh
dsh-tui
```

> 若 `dsh-tui` 命令不存在，等价命令是 `dsh --profile dsh-tui`（两者启动同一个 profile）。

首次启动会走向导（API Key / 语言主题 / 模型工作区 / 快捷键）。API Key 已从环境变量读到的话直接确认即可。

退出 tmux 但保留会话：`Ctrl+B` 然后 `D`；回来：`tmux attach -t dsh`。

---

## 7. 验收清单（必须全部通过）

| # | 检查 | 命令 / 操作 | 期望 |
|---|---|---|---|
| 1 | CLI 版本 | `dsh --version` | `0.2.1-rc.2` |
| 2 | 环境体检 | `dsh-tui doctor` | dsh / pnpm / profile / key 全部 ✓ |
| 3 | profile 为 1017 构建 | 见 4.7 | grep 命中 1017 |
| 4 | 模型列表 | TUI 内输入 `/model`（选 `deepseek-official` 路由） | 出现 `DeepSeek-V4.1-Flash 1017` |
| 5 | 选中并确认 | 选该项后看底部状态栏 | 模型名带 `1017` |
| 6 | 真实调用 | 发一句"用一句话自我介绍" | 正常流式回复，无 401 |
| 7 | 会话落盘 | 退出后重进 `/resume` | 能看到刚才的会话 |

任何一项不通过，按第 10 节排查，**不要继续往下走**。

---

## 8. 日常使用

```bash
tmux attach -t dsh      # 回到 TUI
dsh-tui --resume        # 恢复上次会话
dsh-tui doctor          # 出问题时先跑这个
dsh-tui safe            # 启动崩溃时的只读诊断
dsh-tui version         # 启动器 + profile 版本
```

TUI 内常用：`/model` 切模型、`/resume` 会话列表、`/new` 新会话、`/compact` 压缩上下文、`/settings` 改配置（含 tps 显示）。

### ⚠️ 更新禁令（在 npm 发布新版前）

**不要执行** `/update`、`dsh-tui update`、`npm i -g @deepseek-harness-tui/dsh-tui@latest`。

它们会把 profile 换回 npm 上 10-13 的 0.15.0：1017 标记消失，且 peer 声明只到 `0.2.0-rc.2`，与 `0.2.1-rc.2` 不匹配。

升级只能按第 9 节，从新 commit 重新构建。

---

## 9. 升级与回滚

### 升级（有新 commit 后）

```bash
cd ~/dsh-TUI
git fetch --all --tags
git checkout <新 commit>
git submodule update --init --recursive
pnpm install --frozen-lockfile
pnpm build
npm pack --ignore-scripts

# 备份旧包，再换新包
sudo cp /opt/dsh/dsh-tui-0.15.0-1017.tgz /opt/dsh/dsh-tui-0.15.0-1017.tgz.bak
sudo mv deepseek-harness-tui-dsh-tui-<新版本>.tgz /opt/dsh/dsh-tui-<新版本>-1017.tgz

sudo npm install -g --ignore-scripts /opt/dsh/dsh-tui-<新版本>-1017.tgz
dsh plugin --profile dsh-tui add file:/opt/dsh/dsh-tui-<新版本>-1017.tgz
dsh-tui doctor
```

升级前建议备份 profile：

```bash
cp -a ~/.dsh/profiles/dsh-tui ~/.dsh/profiles/dsh-tui.bak.$(date +%F)
```

### 回滚

```bash
sudo cp /opt/dsh/dsh-tui-0.15.0-1017.tgz.bak /opt/dsh/dsh-tui-0.15.0-1017.tgz
sudo npm install -g --ignore-scripts /opt/dsh/dsh-tui-0.15.0-1017.tgz
dsh plugin --profile dsh-tui add file:/opt/dsh/dsh-tui-0.15.0-1017.tgz
```

---

## 10. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `dsh: command not found` | npm 全局 bin 不在 PATH | `npm prefix -g`，把 `<prefix>/bin` 加进 PATH |
| clone 后 `vendor/dsh-std` 为空 | 没递归拉子模块 | `git submodule update --init --recursive` |
| `pnpm install` 报 lockfile/版本错误 | pnpm 版本不对 | `npm i -g pnpm@11.21.0`，删 `node_modules` 重装 |
| `pnpm build` 卡住/被杀 | 内存不足 | 加 swap（2–4G）或换 4G+ 内存机器 |
| `pnpm build` 仅 verify 失败 | 环境相关校验 | `pnpm compile` 后直接打包 |
| `npm pack` 报 `prepare-guard` | 没加 `--ignore-scripts` | 改用 `npm pack --ignore-scripts` |
| 打包后 tar 里没有 `lib/` | 没先构建 | 先 `pnpm build`（或 `pnpm compile`）再打包 |
| `ERR_PNPM_IGNORED_BUILDS`（编 profile 时） | pnpm ≥11 拦安装脚本 | 按 4.6 写 `allowBuilds` 后重跑 |
| `/model` 里没有 1017 | profile 装成了 npm 0.15.0 | `grep -rl 1017 ...`；重装构建包 |
| 启动即退出、日志说 TTY | 用了管道/重定向 | 进 tmux 或真实终端 |
| `401 Authentication Fails` | key 无效/没进环境 | `echo $DEEPSEEK_API_KEY`；curl balance |
| `~/.dsh` 权限错误 | 曾用 sudo 跑过 | `sudo chown -R "$USER":"$USER" ~/.dsh` |
| 磁盘不足 | 构建 + 依赖峰值 4–6GB | `df -h`，清理旧 `node_modules` 后重试 |

---

## 11. 给 Claude Code 的执行清单

按顺序执行，**每步校验通过再进下一步**；任何一步与期望不符就停下报告，不要"跳过继续"。

1. `uname -m && cat /etc/os-release` — 确认 Linux x86_64/arm64、Ubuntu/Debian
2. `node -v` — 不存在，或版本低于 22.19 且不是 24+ → 按第 2 节装 Node 24
3. `pnpm -v` — 不是 11.21.0 → `sudo npm i -g pnpm@11.21.0`
4. `dsh --version` — 不是 `0.2.1-rc.2` → `sudo npm i -g @deepseek-ai/dsh@0.2.1-rc.2`
5. `git clone --recurse-submodules https://github.com/ccch1mneyyy/dsh-TUI.git`
6. `git checkout e709863019d56897e64495699c3a945cce194d00`
7. `test -f vendor/dsh-std/packages/core/package.json` — 必须存在
8. `pnpm install --frozen-lockfile && pnpm build` — 退出码 0
9. `npm pack --ignore-scripts` — 产出 `deepseek-harness-tui-dsh-tui-0.15.0.tgz`
10. 解包 grep `1017` — 必须输出 `1`
11. 移到 `/opt/dsh/dsh-tui-0.15.0-1017.tgz`
12. `sudo npm i -g --ignore-scripts /opt/dsh/dsh-tui-0.15.0-1017.tgz`
13. `dsh plugin --profile dsh-tui add file:/opt/dsh/dsh-tui-0.15.0-1017.tgz` — 退出码 0
14. `grep -rl 1017 ~/.dsh/profiles/dsh-tui/node_modules/@deepseek-harness-tui/dsh-tui/lib` — 必须有输出
15. `dsh-tui doctor` — 全部 ✓（若只有 config 缺失，首次启动会自动生成）
16. 在 tmux 里启动 `dsh-tui`，`/model` 必须看到 `DeepSeek-V4.1-Flash 1017`
17. 发一句话验证回复正常
18. 汇报：dsh 版本、profile 版本、模型显示名、doctor 结果、tmux 会话名

**禁止项**：
- 不要用 root 启动 TUI
- 不要运行 `dsh-tui update` / `/update` / `@latest`
- 不要删除 `/opt/dsh/*.tgz`
- 不要把 API Key 写进仓库或日志

---

## 附录 A：可选——从现成 tgz 安装（有传输通道时）

如果以后有跳板机、内网文件服务器或 U 盘，能拿到笔记本上这份已构建好的包，可以跳过第 4.1–4.3 节：

```bash
# 上传到服务器 /tmp 后
sha256sum /tmp/dsh-tui-0.15.0-1017.tgz
```

**这份包的校验和**（仅对笔记本上 `/Users/wangye/Documents/deepseek-harness/default-workspace/dsh-TUI/deepseek-harness-tui-dsh-tui-0.15.0.tgz` 有效）：

```text
4164889686becc4f8b4b4b9ac86a71190ad8b0c4ed8e844b26d2c82c2add41c2
```

然后从 4.4 节继续。

---

## 附录 B：目录与文件

```text
~/dsh-TUI/                           # 构建用的源码（装完可保留，便于升级）
/opt/dsh/dsh-tui-0.15.0-1017.tgz     # 安装包（别删）
~/.dsh/                              # DSH_HOME
  ├── profiles/dsh-tui/              # TUI profile（package.json + cordis.patch.yml + node_modules）
  ├── profiles/dsh-tui/cordis.patch.yml   # 用户覆盖层，升级时保留
  └── sessions/                      # 会话库
~/.dsh-tui/settings.json             # TUI 设置（主题/显示/快捷键）
```

---

## 附录 C：安全建议

- 用普通用户运行，TUI 的沙箱与审批策略按用户权限生效
- API Key 只放 `~/.dsh/env`（`chmod 600`），不要进仓库；也不要把 key 直接写在命令行历史里
- 服务器防火墙不需要为 TUI 开放任何端口（纯本地 TTY）
- Key 泄露后立即到 https://platform.deepseek.com/api_keys 删除重建

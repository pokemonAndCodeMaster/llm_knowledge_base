# NotebookLM MCP：从零到精通的保姆级使用指南

本文档将手把手教你如何部署、配置并使用 `notebooklm-mcp`。通过这个工具，你可以让 AI 助手直接连接你私人的 Google NotebookLM 知识库，替你全自动完成“搜集资料 -> 建立笔记本 -> 提问总结 -> 生成播客音频”的完整闭环。

---

## 阶段一：前期部署与安装（仅需一次）

该插件本质上是一个中间服务器，我们需要先把它跑起来。

### 1. 准备工作
确保你的电脑（或 WSL 环境）已经安装了 **Node.js** (推荐 v18 及以上版本)。

### 2. 下载并安装
在你打算存放项目的目录下（例如你的代码仓目录），打开终端，依次运行以下三条命令：

```bash
# 1. 下载源代码
git clone https://github.com/jackc1111/antigravity-notebooklm-mcp notebooklm-mcp

# 2. 进入文件夹
cd notebooklm-mcp

# 3. 安装依赖并编译构建
npm install && npm run build
```
*注：如果在安装过程中看到一些黄色的 `WARN`（警告），不用担心，只要没有红色的 `ERR!`（错误）就代表安装成功。*

---

## 阶段二：打通连接与鉴权（最核心的一步）

因为 NotebookLM 是 Google 的产品，它有非常变态的机器检测机制。如果让机器自己去输入密码登录，100% 会被封锁。
因此，我们需要用到**“偷渡法”（手动抓取 Cookie 注入）**。

### 1. 傻瓜式抓取 Cookie
1. 打开你平时用来上网的电脑浏览器（推荐 Chrome 或 Edge）。
2. 在浏览器中打开并登录 [Google NotebookLM 官网](https://notebooklm.google.com/)。
3. 登录成功，看到你的笔记本列表后，按下键盘上的 **F12** 键，打开“开发者工具”。
4. 在开发者工具中，点击顶部的 **Network（网络）** 标签页。
5. **按 F5 刷新一下整个网页。**
6. 这时候 Network 面板里会出现密密麻麻的请求，在左侧列表随便点击前几个请求（通常找名字类似 `list`、`notebooks` 或干脆随便点一个）。
7. 在右侧弹出的详情中，找到 **Headers（标头）** -> 往下滚找到 **Request Headers（请求标头）**。
8. 找到里面名叫 `Cookie:` 的那一行。
9. **右键点击 Cookie 的值 -> 选择 Copy Value (复制值)**。此时一长串类似天书的乱码就存进了你的剪贴板。

### 2. 注入 Cookie 让系统记住你
在你的终端里新建一个名叫 `save_auth.py` 的脚本文件，用来把那堆乱码解析好存起来：

```bash
# 确保你现在还在 notebooklm-mcp 文件夹里
cat > save_auth.py << 'EOF'
import json, os
# 👇 把你刚刚复制的那一大段天书，粘贴替换掉下面引号里的内容 👇
cookie_str = "把刚才复制的超长Cookie粘贴到这里，记得保留两边的双引号"

cookies = {}
for item in cookie_str.split(';'):
    if '=' in item:
        key, value = item.strip().split('=', 1)
        cookies[key] = value

os.makedirs(os.path.expanduser("~/.notebooklm-mcp"), exist_ok=True)
auth_path = os.path.expanduser("~/.notebooklm-mcp/auth.json")

with open(auth_path, "w") as f:
    json.dump({"cookies": cookies, "csrf_token": ""}, f, indent=2)
print("恭喜！鉴权文件已成功保存到", auth_path)
EOF
```

编辑好粘贴进去后，运行它：
```bash
python3 save_auth.py
```
*（如果输出“恭喜！鉴权文件已成功保存...”，就说明配置完成！以后只有当你发现 AI 连不上 Notebook 时，才需要重复上述步骤更新一下 Cookie。）*

---

## 阶段三：将插件接入到 AI 客户端

现在我们的 MCP 服务器已经准备就绪，只要把它告诉能支持 MCP 的 AI 软件（比如 Claude Desktop、Cursor 或 Cline）即可。

以 **Claude Desktop** 为例：
1. 找到 Claude 的配置文件：
   - Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
2. 使用文本编辑器打开它，加上我们刚才编译好的文件路径：

```json
{
  "mcpServers": {
    "notebooklm": {
      "command": "node",
      "args": [
        "C:\\你的实际路径\\notebooklm-mcp\\build\\index.js"  // Linux/Mac 请换成绝对路径如 /home/user/.../build/index.js
      ]
    }
  }
}
```
保存文件，**彻底重启 Claude Desktop**，你会看到它多了一把类似“锤子”的工具图标，说明连接成功！

---

## 阶段四：日常使用与人机协同指南

只要配置好，你以后**完全不需要自己去点网页，也不需要敲任何代码**。你需要做的仅仅是对着 AI 下达“大白话”指令。

以下为你提供几个高频的傻瓜式提问模板，你可以直接复制粘贴发给 AI（比如发给现在的我）：

### 💡 场景 1：你想偷懒，让 AI 自己去全网搜集资料并建库
> **怎么对我说：**
> “帮我在 NotebookLM 里新建一个名叫『AI大模型推理架构』的笔记本，然后针对『DeepSeek V3 架构原理与推理优化』做深度研究，把你找到的有价值的网页全自动导入进这个笔记本里。”

**AI 会怎么做：** 我会自动建好笔记本，启动后台的爬虫引擎全网搜索论文和博客，帮你精选后一条条存进这个本子里。你泡杯茶回来，资料库就建好了。

### 💡 场景 2：你想针对已有的知识库进行硬核拷问
> **怎么对我说：**
> “在我已有的『VLM架构演进与自动驾驶端到端训练全景指南』这个笔记本里查一下，主流公司是怎么解决 3D 视觉特征丢失问题的？请根据笔记本里的资料给我一份总结。”

**AI 会怎么做：** 我会精准定位到你说的这个笔记本，将问题直接发给 Google 的底层引擎，利用 NotebookLM 强大的 RAG（检索增强生成）能力，把散落在多篇论文中的答案揉碎了总结给你听。

### 💡 场景 3：你想自动管理资料
> **怎么对我说：**
> “查看一下我现在的账号里总共有哪些笔记本？如果有名字带有『example』或者『测试』的，直接帮我全删了。”

**AI 会怎么做：** 我会列出你的清单，根据你的规则帮你执行清理，省去了你逐个点开网页删除的麻烦。

### 💡 场景 4：把长篇大论变成轻松的播客音频
> **怎么对我说：**
> “针对我名叫『多模态大模型』的笔记本，帮我生成一份音频播客总结（Audio Overview），生成完告诉我。”

**AI 会怎么做：** 我会自动向 Google 下发生成指令，后台的两人英文播客就开始自动生成了。几分钟后你去官网，就能直接戴上耳机听书了。

---

**最终结语：**
在这个工作流中，你扮演的是**司令员**，我是**参谋长**，而 MCP 是我的**对讲机**。
你只需负责下达高层逻辑指令（“要查什么”、“要总结什么”），剩下的工具调用、参数匹配、数据搬运，全部由我来负责！

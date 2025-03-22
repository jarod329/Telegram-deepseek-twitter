# 推特自动发文MCP服务使用和部署指南

## 服务简介

这个MCP（微服务控制面板）服务可以自动从Telegram频道获取加密货币相关消息，使用AI生成内容，并发布到Twitter上。服务通过RESTful API提供，支持多用户，每个用户可以配置自己的API密钥和发布设置。

## 部署方式

### 方法一：传统服务器部署

1. **环境准备**

```bash
# 克隆代码到服务器
git clone https://github.com/your-username/twitter-poster-mcp.git
cd twitter-poster-mcp

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

2. **配置requirements.txt文件**（确保包含以下依赖）

```
fastapi>=0.104.0
uvicorn>=0.23.2
tweepy>=4.14.0
openai>=1.0.0
telethon>=1.30.0
httpx>=0.24.1
schedule>=1.2.0
pydantic>=2.4.2
python-dotenv>=1.0.0
```

3. **启动服务**

```bash
# 基本启动
python main.py

# 指定主机和端口
python main.py --host 0.0.0.0 --port 8080

# 开发模式（自动重载）
python main.py --reload
```

### 方法二：使用Docker部署

1. **创建Dockerfile**

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
```

2. **创建docker-compose.yml**（可选）

```yaml
version: '3'

services:
  twitter-poster-mcp:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./configs:/app/configs
    restart: unless-stopped
    environment:
      - TZ=Asia/Shanghai
```

3. **构建和运行**

```bash
# 使用Dockerfile构建
docker build -t twitter-poster-mcp .
docker run -p 8000:8000 -v $(pwd)/configs:/app/configs twitter-poster-mcp

# 或使用docker-compose
docker-compose up -d
```

### 方法三：使用云服务部署

1. **AWS Elastic Beanstalk**
   - 创建一个Python应用
   - 上传代码包
   - 设置环境变量

2. **Heroku**
   - 创建Procfile: `web: uvicorn main:app --host=0.0.0.0 --port=$PORT`
   - 使用Heroku CLI部署: `heroku create && git push heroku main`

3. **阿里云/腾讯云**
   - 使用云服务器ECS/CVM
   - 按照传统服务器部署步骤操作
   - 或使用容器服务ACK/TKE部署Docker版本

## 如何使用服务

### 1. 注册服务

首先，用户需要注册并获取API密钥。

```bash
curl -X POST "http://your-server:8000/users/register" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your-preferred-api-key",
    "service_config": {
      "twitter": {
        "consumer_key": "your-twitter-consumer-key",
        "consumer_secret": "your-twitter-consumer-secret",
        "access_token": "your-twitter-access-token",
        "access_token_secret": "your-twitter-access-token-secret"
      },
      "telegram": {
        "api_id": 123456,
        "api_hash": "your-telegram-api-hash",
        "channel": "channel_name"
      },
      "ai": {
        "api_key": "your-deepseek-api-key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
      },
      "prompt_template": "请根据以下从加密货币新闻频道获取的消息，提炼总结成英文推文。若有多条消息，用数字1,2,3等分开。推文要足够吸引人，使用加密货币领域的专业术语，可以添加emoji表情。总字数不超过70个英文单词。原始消息:\n{messages}",
      "fixed_hashtags": "#crypto #bitcoin"
    }
  }'
```

服务将返回一个包含`user_id`和`api_key`的响应。请保存此信息用于后续API调用。

### 2. 使用API

所有后续请求都需要包含API密钥作为Authorization header：

```
Authorization: Bearer your-api-key
```

#### 发送单条推文

```bash
curl -X POST "http://your-server:8000/twitter/post" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Bitcoin just broke $80,000! This is a historic moment for cryptocurrency. #BTC",
    "hashtags": "#crypto #bitcoin #btc"
  }'
```

#### 获取Telegram最新消息

```bash
curl -X POST "http://your-server:8000/telegram/fetch" \
  -H "Authorization: Bearer your-api-key"
```

#### 生成推文内容

```bash
curl -X POST "http://your-server:8000/generate/content" \
  -H "Authorization: Bearer your-api-key"
```

#### 启动自动发布定时任务

```bash
curl -X POST "http://your-server:8000/scheduler/start" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "interval_minutes": 30,
    "min_interval_seconds": 2400,
    "max_interval_seconds": 4800
  }'
```

#### 停止自动发布

```bash
curl -X POST "http://your-server:8000/scheduler/stop" \
  -H "Authorization: Bearer your-api-key"
```

#### 获取服务状态

```bash
curl -X GET "http://your-server:8000/status" \
  -H "Authorization: Bearer your-api-key"
```

#### 更新配置

```bash
curl -X PUT "http://your-server:8000/config/update" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "twitter": {
      "consumer_key": "updated-key",
      "consumer_secret": "updated-secret",
      "access_token": "updated-token",
      "access_token_secret": "updated-token-secret"
    },
    "telegram": {
      "api_id": 123456,
      "api_hash": "your-telegram-api-hash",
      "channel": "new_channel_name"
    },
    "ai": {
      "api_key": "your-deepseek-api-key"
    },
    "fixed_hashtags": "#crypto #bitcoin #eth"
  }'
```

#### 注销用户

```bash
curl -X DELETE "http://your-server:8000/users/unregister" \
  -H "Authorization: Bearer your-api-key"
```

## 其他开发者可以如何接入这个服务

其他开发者可以通过以下方式接入此服务：

1. **Web应用集成**
   - 在前端应用中通过API请求调用相关功能
   - 创建图形界面来管理Twitter自动发文

2. **移动应用集成**
   - 在移动应用中通过API请求调用相关功能
   - 提供推文预览和管理界面

3. **其他自动化系统集成**
   - 在自己的自动化流程中调用此API
   - 例如：将新闻监控系统与此服务组合

4. **第三方平台集成**
   - 将此服务与其他平台（如WordPress、Discord机器人等）集成

## 安全建议

1. 使用HTTPS保护API通信
2. 设置复杂的API密钥
3. 定期更新Twitter和Telegram的API凭据
4. 监控日志以检测异常活动
5. 备份配置文件

## 常见问题

1. **Q: API返回401错误**
   A: 检查API密钥是否正确，并确保在请求头中正确设置Authorization

2. **Q: 无法连接到Telegram**
   A: 检查网络连接、代理设置和Telegram API凭据

3. **Q: 推文发送失败**
   A: 检查Twitter API凭据和限制状态，可能达到推文频率限制

4. **Q: 服务占用内存过高**
   A: 考虑限制同时运行的实例数量或增加服务器资源

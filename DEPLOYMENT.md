# 项目部署指南（不备案方案）

## 目录
1. [准备工作](#准备工作)
2. [服务器购买与配置](#服务器购买与配置)
3. [域名注册与解析](#域名注册与解析)
4. [项目部署](#项目部署)
5. [防火墙配置](#防火墙配置)
6. [启动服务](#启动服务)
7. [访问应用](#访问应用)

---

## 准备工作

### 选择服务器
- 推荐：雨云服务器（Ubuntu 20.04/22.04）
- 配置建议：2核4G 以上（YOLOv8 模型需要较多内存）
- 系统：Ubuntu 20.04 LTS 或 22.04 LTS

### 选择免费域名
- Freenom（.tk、.ml、.ga、.cf、.gq）
- EU.org
- No-IP（动态DNS）
- 或使用其他免费域名服务

---

## 服务器购买与配置

### 1. 购买雨云服务器
1. 访问 [雨云官网](https://www.rainyun.com/)
2. 注册账号并登录
3. 选择云服务器产品
4. 配置选择：
   - 地域：选择离你较近的区域
   - 配置：2核4G以上
   - 系统：Ubuntu 20.04 LTS
   - 带宽：3Mbps以上
5. 购买并获取服务器信息（IP地址、root密码）

### 2. 登录服务器
使用 SSH 工具（如 Xshell、PuTTY、终端）登录服务器：

```bash
ssh root@你的服务器IP
```

---

## 域名注册与解析

### 1. 注册免费域名
以 Freenom 为例：
1. 访问 [Freenom](https://www.freenom.com/)
2. 搜索你想要的域名（如 example.tk）
3. 选择免费域名并注册
4. 完成注册流程

### 2. 域名解析
1. 登录域名管理后台
2. 添加 DNS 解析记录：
   - 记录类型：A 记录
   - 主机记录：@ 或 www
   - 记录值：你的服务器IP
   - TTL：默认即可

---

## 项目部署

### 1. 服务器环境配置

#### 更新系统
```bash
apt update && apt upgrade -y
```

#### 安装 Python 3 和 pip
```bash
apt install -y python3 python3-pip python3-venv
```

#### 安装必要工具
```bash
apt install -y git nginx supervisor
```

### 2. 上传项目代码

#### 方式一：Git 克隆（推荐）
```bash
cd /opt
git clone <你的项目仓库地址>
cd basemenufoecheck
```

#### 方式二：SCP 上传
在本地终端执行：
```bash
scp -r /path/to/your/project root@你的服务器IP:/opt/
```

### 3. 创建虚拟环境
```bash
cd /opt/basemenufoecheck
python3 -m venv venv
source venv/bin/activate
```

### 4. 安装依赖
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install flask flask-cors gunicorn
```

---

## 防火墙配置

由于不备案，我们使用非标准端口（如 8080、8443）

```bash
# 配置 UFW 防火墙
ufw allow 22/tcp    # SSH
ufw allow 8080/tcp  # 应用端口
ufw enable
```

---

## 启动服务

### 1. 创建必要目录
```bash
mkdir -p /opt/basemenufoecheck/uploads
mkdir -p /opt/basemenufoecheck/exports
chmod 755 /opt/basemenufoecheck/uploads
chmod 755 /opt/basemenufoecheck/exports
```

### 2. 使用 Gunicorn 启动（生产环境）

#### 创建 Gunicorn 配置文件
```bash
cat > /opt/basemenufoecheck/gunicorn_config.py << 'EOF'
import multiprocessing

bind = "0.0.0.0:8080"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 5
accesslog = "/opt/basemenufoecheck/logs/access.log"
errorlog = "/opt/basemenufoecheck/logs/error.log"
loglevel = "info"
EOF
```

#### 创建日志目录
```bash
mkdir -p /opt/basemenufoecheck/logs
```

### 3. 使用 Supervisor 管理进程

#### 创建 Supervisor 配置
```bash
cat > /etc/supervisor/conf.d/imageforai.conf << 'EOF'
[program:imageforai]
command=/opt/basemenufoecheck/venv/bin/gunicorn -c /opt/basemenufoecheck/gunicorn_config.py app:app
directory=/opt/basemenufoecheck
user=root
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/opt/basemenufoecheck/logs/supervisor.log
environment=LANG="en_US.UTF-8",LC_ALL="en_US.UTF-8"
EOF
```

#### 启动 Supervisor
```bash
supervisorctl reread
supervisorctl update
supervisorctl start imageforai
```

#### 查看服务状态
```bash
supervisorctl status imageforai
```

---

## 访问应用

部署完成后，通过以下地址访问：
```
http://你的域名:8080
或
http://你的服务器IP:8080
```

---

## 常见问题

### 1. 服务无法启动
```bash
# 查看日志
tail -f /opt/basemenufoecheck/logs/error.log
tail -f /opt/basemenufoecheck/logs/supervisor.log
```

### 2. 重启服务
```bash
supervisorctl restart imageforai
```

### 3. 停止服务
```bash
supervisorctl stop imageforai
```

---

## 后续优化（可选）

### 使用 Nginx 反向代理
如果需要更好的性能，可以配置 Nginx：

```nginx
server {
    listen 8080;
    server_name 你的域名;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 注意事项
- 不备案无法使用 80/443 端口
- 建议使用 CDN 加速（部分 CDN 支持非标准端口）
- 定期备份数据
- 监控服务器资源使用情况

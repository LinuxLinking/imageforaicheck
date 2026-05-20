#!/bin/bash

# AI图像检测工具 - 一键部署脚本
# 适用于 Ubuntu 20.04/22.04
# 使用非标准端口（8080），无需备案

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目配置
PROJECT_DIR="/opt/basemenufoecheck"
PROJECT_NAME="imageforai"
PORT=8080

# 打印信息函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为 root 用户
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "请使用 root 用户运行此脚本"
        exit 1
    fi
}

# 更新系统
update_system() {
    print_info "正在更新系统..."
    apt update && apt upgrade -y
}

# 安装依赖
install_dependencies() {
    print_info "正在安装系统依赖..."
    apt install -y python3 python3-pip python3-venv git nginx supervisor
}

# 检查项目目录
check_project_dir() {
    if [ ! -d "$PROJECT_DIR" ]; then
        print_error "项目目录不存在: $PROJECT_DIR"
        print_info "请先将项目代码上传到 $PROJECT_DIR"
        exit 1
    fi
}

# 创建虚拟环境
create_venv() {
    print_info "正在创建 Python 虚拟环境..."
    cd "$PROJECT_DIR"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
}

# 安装 Python 依赖
install_python_deps() {
    print_info "正在安装 Python 依赖..."
    cd "$PROJECT_DIR"
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r backend/requirements.txt
    pip install flask flask-cors gunicorn
}

# 创建必要目录
create_dirs() {
    print_info "正在创建必要目录..."
    cd "$PROJECT_DIR"
    mkdir -p uploads exports logs
    chmod 755 uploads exports logs
}

# 创建 Gunicorn 配置
create_gunicorn_config() {
    print_info "正在创建 Gunicorn 配置..."
    cat > "$PROJECT_DIR/gunicorn_config.py" << 'EOF'
import multiprocessing

bind = "0.0.0.0:8080"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 300
keepalive = 5
accesslog = "/opt/basemenufoecheck/logs/access.log"
errorlog = "/opt/basemenufoecheck/logs/error.log"
loglevel = "info"
EOF
}

# 创建 Supervisor 配置
create_supervisor_config() {
    print_info "正在创建 Supervisor 配置..."
    cat > "/etc/supervisor/conf.d/$PROJECT_NAME.conf" << EOF
[program:$PROJECT_NAME]
command=$PROJECT_DIR/venv/bin/gunicorn -c $PROJECT_DIR/gunicorn_config.py app:app
directory=$PROJECT_DIR
user=root
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=$PROJECT_DIR/logs/supervisor.log
environment=LANG="en_US.UTF-8",LC_ALL="en_US.UTF-8"
EOF
}

# 配置防火墙
configure_firewall() {
    print_info "正在配置防火墙..."
    ufw allow 22/tcp
    ufw allow $PORT/tcp
    ufw --force enable
}

# 启动服务
start_service() {
    print_info "正在启动服务..."
    supervisorctl reread
    supervisorctl update
    supervisorctl start $PROJECT_NAME
}

# 显示服务状态
show_status() {
    print_info "服务状态:"
    supervisorctl status $PROJECT_NAME
}

# 获取服务器 IP
get_server_ip() {
    local ip=$(hostname -I | awk '{print $1}')
    echo $ip
}

# 显示完成信息
show_completion() {
    local ip=$(get_server_ip)
    print_info "========================================"
    print_info "部署完成！"
    print_info "========================================"
    echo ""
    print_info "访问地址:"
    print_info "  http://$ip:$PORT"
    print_info "  http://你的域名:$PORT"
    echo ""
    print_info "常用命令:"
    print_info "  查看状态: supervisorctl status $PROJECT_NAME"
    print_info "  查看日志: tail -f $PROJECT_DIR/logs/error.log"
    print_info "  重启服务: supervisorctl restart $PROJECT_NAME"
    print_info "  停止服务: supervisorctl stop $PROJECT_NAME"
    echo ""
}

# 主函数
main() {
    print_info "========================================"
    print_info "AI图像检测工具 - 一键部署"
    print_info "========================================"
    echo ""

    check_root
    check_project_dir
    update_system
    install_dependencies
    create_venv
    install_python_deps
    create_dirs
    create_gunicorn_config
    create_supervisor_config
    configure_firewall
    start_service
    show_status
    show_completion
}

# 运行主函数
main "$@"

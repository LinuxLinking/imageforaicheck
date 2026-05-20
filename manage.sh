#!/bin/bash

# AI图像检测工具 - 服务管理脚本
# 用于管理部署后的服务

set -e

# 配置
PROJECT_DIR="/opt/basemenufoecheck"
PROJECT_NAME="imageforai"
PORT=8080

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为 root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "请使用 root 用户运行此脚本"
        exit 1
    fi
}

# 显示帮助
show_help() {
    echo "AI图像检测工具 - 服务管理脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  start     - 启动服务"
    echo "  stop      - 停止服务"
    echo "  restart   - 重启服务"
    echo "  status    - 查看服务状态"
    echo "  logs      - 查看日志"
    echo "  logs-error - 查看错误日志"
    echo "  logs-access - 查看访问日志"
    echo "  update    - 更新代码并重启"
    echo "  help      - 显示此帮助信息"
    echo ""
}

# 启动服务
start_service() {
    print_info "正在启动服务..."
    supervisorctl start $PROJECT_NAME
    print_info "服务已启动"
}

# 停止服务
stop_service() {
    print_info "正在停止服务..."
    supervisorctl stop $PROJECT_NAME
    print_info "服务已停止"
}

# 重启服务
restart_service() {
    print_info "正在重启服务..."
    supervisorctl restart $PROJECT_NAME
    print_info "服务已重启"
}

# 查看状态
status_service() {
    print_info "服务状态:"
    supervisorctl status $PROJECT_NAME
}

# 查看日志
view_logs() {
    print_info "查看 Supervisor 日志 (按 Ctrl+C 退出):"
    tail -f "$PROJECT_DIR/logs/supervisor.log"
}

# 查看错误日志
view_error_logs() {
    print_info "查看错误日志 (按 Ctrl+C 退出):"
    tail -f "$PROJECT_DIR/logs/error.log"
}

# 查看访问日志
view_access_logs() {
    print_info "查看访问日志 (按 Ctrl+C 退出):"
    tail -f "$PROJECT_DIR/logs/access.log"
}

# 更新代码
update_code() {
    print_info "正在更新代码..."
    cd "$PROJECT_DIR"
    
    if [ -d ".git" ]; then
        git pull
        print_info "代码已更新"
        
        print_info "正在重启服务..."
        restart_service
    else
        print_warning "不是 Git 仓库，无法自动更新"
        print_info "请手动更新代码后运行: $0 restart"
    fi
}

# 主函数
main() {
    check_root
    
    case "${1:-help}" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        status)
            status_service
            ;;
        logs)
            view_logs
            ;;
        logs-error)
            view_error_logs
            ;;
        logs-access)
            view_access_logs
            ;;
        update)
            update_code
            ;;
        help)
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"

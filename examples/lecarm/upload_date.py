import paramiko
import os
import time
from typing import Optional, List, Dict, Callable, Union, Any
import hashlib
import sys
from pathlib import Path

class UnifiedSSHClient:
    """
    统一的SSH客户端 - 同时支持文件传输和命令执行
    """
    
    def __init__(self, 
                 host: str = "region-9.autodl.pro",
                 port: int = 49476,
                 username: str = "root",
                 password: str = "sCSRzjrEPVeT"):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.ssh_client = None
        self.sftp_client = None
        self.connected = False
    
    def connect(self) -> bool:
        """建立SSH和SFTP连接"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 连接SSH
            self.ssh_client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=10,
                banner_timeout=20,
                auth_timeout=10
            )
            
            # 连接SFTP
            self.sftp_client = self.ssh_client.open_sftp()
            
            self.connected = True
            print(f"✅ 已连接到 {self.username}@{self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开连接"""
        if self.sftp_client:
            self.sftp_client.close()
        if self.ssh_client:
            self.ssh_client.close()
        self.connected = False
        print("🔌 连接已关闭")
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            if not self.connected:
                return self.connect()
            
            # 执行一个简单命令测试
            result = self.execute_command("echo 'test'")
            return result.get('success', False)
            
        except Exception as e:
            print(f"❌ 连接测试失败: {e}")
            return False
    
    # ==================== 文件传输功能 ====================
    
    def upload_file(self, 
                   local_path: str, 
                   remote_path: str,
                   callback: Optional[Callable] = None,
                   overwrite: bool = False) -> Dict:
        """
        上传单个文件
        
        参数:
            local_path: 本地文件路径
            remote_path: 远程文件路径
            callback: 进度回调函数 callback(transferred, total)
            overwrite: 是否覆盖已存在文件
        """
        if not self.connected:
            print("❌ 未连接到服务器")
            return {'success': False, 'error': '未连接'}
        
        try:
            if not os.path.exists(local_path):
                print(f"❌ 本地文件不存在: {local_path}")
                return {'success': False, 'error': '本地文件不存在'}
            
            # 确保远程目录存在
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self._ensure_remote_directory(remote_dir)
            
            # 检查远程文件是否已存在
            try:
                self.sftp_client.stat(remote_path)
                if not overwrite:
                    print(f"⚠️ 远程文件已存在: {remote_path}")
                    return {'success': False, 'error': '文件已存在'}
            except FileNotFoundError:
                pass
            
            # 获取文件大小
            file_size = os.path.getsize(local_path)
            print(f"📤 上传: {os.path.basename(local_path)} ({self._format_size(file_size)})")
            
            # 包装回调函数
            def wrapped_callback(transferred, to_be_transferred):
                if callback:
                    callback(transferred, file_size)
            
            # 执行上传
            self.sftp_client.put(local_path, remote_path, 
                               callback=wrapped_callback if callback else None)
            
            print(f"✅ 上传成功: {local_path} -> {remote_path}")
            return {
                'success': True,
                'filename': os.path.basename(local_path),
                'size': file_size,
                'local_path': local_path,
                'remote_path': remote_path
            }
            
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def upload_directory(self, 
                        local_dir: str, 
                        remote_dir: str,
                        exclude_patterns: Optional[List[str]] = None,
                        callback: Optional[Callable] = None) -> Dict:
        """
        上传整个目录
        
        参数:
            exclude_patterns: 排除的文件模式，如 ['.git', '*.tmp']
            callback: 进度回调 callback(filename, transferred, total)
        """
        if not self.connected:
            print("❌ 未连接到服务器")
            return {'success': False, 'error': '未连接'}
        
        from fnmatch import fnmatch
        
        exclude_patterns = exclude_patterns or []
        results = {
            'success': True,
            'total_files': 0,
            'success_files': 0,
            'failed_files': 0,
            'total_size': 0,
            'details': []
        }
        
        # 确保远程目录存在
        self._ensure_remote_directory(remote_dir)
        
        for root, dirs, files in os.walk(local_dir):
            # 计算相对路径
            rel_path = os.path.relpath(root, local_dir)
            if rel_path == '.':
                current_remote = remote_dir
            else:
                current_remote = os.path.join(remote_dir, rel_path)
            
            # 确保远程子目录存在
            if rel_path != '.':
                self._ensure_remote_directory(current_remote)
            
            # 上传文件
            for filename in files:
                # 检查排除模式
                skip = False
                for pattern in exclude_patterns:
                    if fnmatch(filename, pattern):
                        skip = True
                        break
                
                if skip:
                    continue
                
                local_file = os.path.join(root, filename)
                remote_file = os.path.join(current_remote, filename)
                
                results['total_files'] += 1
                file_size = os.path.getsize(local_file)
                results['total_size'] += file_size
                
                if callback:
                    callback(filename, 0, file_size)
                
                try:
                    # 上传单个文件
                    result = self.upload_file(local_file, remote_file, overwrite=True)
                    
                    if result['success']:
                        results['success_files'] += 1
                        results['details'].append({
                            'file': filename,
                            'status': 'success',
                            'size': file_size
                        })
                    else:
                        results['success'] = False
                        results['failed_files'] += 1
                        results['details'].append({
                            'file': filename,
                            'status': 'failed',
                            'error': result.get('error')
                        })
                    
                    if callback:
                        callback(filename, file_size, file_size)
                        
                except Exception as e:
                    results['success'] = False
                    results['failed_files'] += 1
                    results['details'].append({
                        'file': filename,
                        'status': 'error',
                        'error': str(e)
                    })
        
        print(f"📊 上传完成: {results['success_files']}/{results['total_files']} 文件")
        return results
    
    def download_file(self, 
                     remote_path: str, 
                     local_path: str,
                     callback: Optional[Callable] = None,
                     overwrite: bool = False) -> Dict:
        """
        下载单个文件
        """
        if not self.connected:
            print("❌ 未连接到服务器")
            return {'success': False, 'error': '未连接'}
        
        try:
            # 检查远程文件是否存在
            try:
                remote_stat = self.sftp_client.stat(remote_path)
                file_size = remote_stat.st_size
            except FileNotFoundError:
                print(f"❌ 远程文件不存在: {remote_path}")
                return {'success': False, 'error': '远程文件不存在'}
            
            # 检查本地文件是否已存在
            if os.path.exists(local_path) and not overwrite:
                print(f"⚠️ 本地文件已存在: {local_path}")
                return {'success': False, 'error': '文件已存在'}
            
            # 确保本地目录存在
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            
            print(f"📥 下载: {os.path.basename(remote_path)} ({self._format_size(file_size)})")
            
            # 包装回调函数
            def wrapped_callback(transferred, to_be_transferred):
                if callback:
                    callback(transferred, file_size)
            
            # 执行下载
            self.sftp_client.get(remote_path, local_path,
                               callback=wrapped_callback if callback else None)
            
            print(f"✅ 下载成功: {remote_path} -> {local_path}")
            return {
                'success': True,
                'filename': os.path.basename(remote_path),
                'size': file_size,
                'remote_path': remote_path,
                'local_path': local_path
            }
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_directory(self, remote_path: str = '.') -> List[Dict]:
        """列出远程目录内容"""
        if not self.connected:
            print("❌ 未连接到服务器")
            return []
        
        try:
            items = self.sftp_client.listdir_attr(remote_path)
            result = []
            
            for item in items:
                from stat import S_ISDIR, S_ISREG, S_ISLNK
                
                file_type = 'directory' if S_ISDIR(item.st_mode) else 'file'
                if S_ISLNK(item.st_mode):
                    file_type = 'link'
                
                result.append({
                    'name': item.filename,
                    'type': file_type,
                    'size': item.st_size,
                    'permissions': oct(item.st_mode)[-3:],
                    'modified': time.strftime('%Y-%m-%d %H:%M:%S', 
                                            time.localtime(item.st_mtime))
                })
            
            return result
            
        except Exception as e:
            print(f"❌ 列出目录失败: {e}")
            return []
    
    def delete_remote(self, remote_path: str, recursive: bool = False) -> bool:
        """删除远程文件或目录"""
        if not self.connected:
            print("❌ 未连接到服务器")
            return False
        
        try:
            if recursive:
                # 递归删除目录
                for item in self.sftp_client.listdir(remote_path):
                    item_path = os.path.join(remote_path, item)
                    try:
                        self.sftp_client.remove(item_path)
                    except:
                        self.delete_remote(item_path, True)
                
                self.sftp_client.rmdir(remote_path)
            else:
                # 尝试删除文件
                try:
                    self.sftp_client.remove(remote_path)
                except:
                    # 如果是目录，尝试删除
                    self.sftp_client.rmdir(remote_path)
            
            print(f"✅ 删除成功: {remote_path}")
            return True
            
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    
    def _ensure_remote_directory(self, remote_dir: str) -> None:
        """确保远程目录存在"""
        if not remote_dir or remote_dir == '/':
            return
        
        remote_dir = remote_dir.rstrip('/')
        
        if remote_dir.startswith('/'):
            parts = remote_dir.split('/')[1:]
            current = '/'
        else:
            parts = remote_dir.split('/')
            current = ''
        
        for part in parts:
            if not part:
                continue
            
            if current == '/':
                next_dir = f"/{part}"
            elif current:
                next_dir = f"{current}/{part}"
            else:
                next_dir = part
            
            try:
                self.sftp_client.stat(next_dir)
            except FileNotFoundError:
                try:
                    self.sftp_client.mkdir(next_dir)
                except Exception as e:
                    print(f"创建目录失败 {next_dir}: {e}")
                    # 尝试通过SSH命令创建
                    self.execute_command(f"mkdir -p {next_dir}", show_output=False)
            
            current = next_dir
    
    # ==================== 命令执行功能 ====================
    
    def execute_command(self, 
                       command: str, 
                       timeout: int = 30,
                       show_output: bool = True,
                       capture_output: bool = True) -> Dict:
        """
        执行SSH命令
        
        参数:
            command: 要执行的命令
            timeout: 超时时间（秒）
            show_output: 是否显示实时输出
            capture_output: 是否捕获输出
            
        返回:
            命令执行结果字典
        """
        if not self.connected:
            print("❌ 未连接到服务器")
            return {'success': False, 'error': '未连接'}
        
        print(f"🚀 执行: {command}")
        
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command, timeout=timeout)
            
            # 实时读取输出
            output_lines = []
            error_lines = []
            
            # 读取标准输出
            while True:
                line = stdout.readline()
                if not line:
                    break
                if show_output:
                    print(line.rstrip())
                if capture_output:
                    output_lines.append(line)
            
            # 读取标准错误
            while True:
                line = stderr.readline()
                if not line:
                    break
                if show_output:
                    print(f"ERROR: {line.rstrip()}", file=sys.stderr)
                if capture_output:
                    error_lines.append(line)
            
            # 获取退出码
            exit_code = stdout.channel.recv_exit_status()
            
            result = {
                'success': exit_code == 0,
                'exit_code': exit_code,
                'command': command
            }
            
            if capture_output:
                result['stdout'] = ''.join(output_lines)
                result['stderr'] = ''.join(error_lines)
            
            if exit_code != 0:
                print(f"⚠️ 命令退出码: {exit_code}")
            
            return result
            
        except Exception as e:
            error_msg = f"❌ 命令执行失败: {e}"
            print(error_msg)
            return {'success': False, 'error': error_msg, 'command': command}
    
    def execute_multiple_commands(self, 
                                 commands: List[str],
                                 timeout: int = 30,
                                 show_output: bool = True) -> List[Dict]:
        """
        执行多个SSH命令
        """
        results = []
        
        for cmd in commands:
            print(f"\n{'='*50}")
            result = self.execute_command(cmd, timeout, show_output, capture_output=True)
            results.append(result)
        
        return results
    
    def interactive_shell(self) -> None:
        """
        启动交互式Shell
        """
        if not self.connected:
            print("❌ 未连接到服务器")
            return
        
        channel = self.ssh_client.invoke_shell()
        channel.settimeout(0.0)  # 非阻塞模式
        
        print("\n" + "="*60)
        print("🖥️  SSH交互式终端")
        print("输入 'exit' 或按 Ctrl+D 退出")
        print("="*60 + "\n")
        
        # 清空初始缓冲区
        self._read_channel(channel, 1.0)
        
        try:
            while True:
                # 读取远程输出
                output = self._read_channel(channel, 0.1)
                if output:
                    print(output, end='', flush=True)
                
                # 检查是否有本地输入
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline()
                    
                    if not user_input:  # Ctrl+D
                        print("\n👋 退出")
                        break
                    
                    if user_input.strip().lower() in ['exit', 'quit']:
                        channel.send("exit\n")
                        time.sleep(0.5)
                        break
                    
                    # 发送命令
                    channel.send(user_input)
                
                time.sleep(0.05)
        
        except KeyboardInterrupt:
            print("\n\n👋 中断连接")
        except ImportError:
            # Windows没有select模块，使用简化版
            self._simple_interactive_shell(channel)
        finally:
            channel.close()
    
    def _simple_interactive_shell(self, channel) -> None:
        """简化的交互式Shell（用于Windows）"""
        print("⚠️  Windows简化交互模式，输入'exit'退出")
        
        try:
            while True:
                # 读取远程输出
                if channel.recv_ready():
                    data = channel.recv(1024).decode('utf-8', errors='ignore')
                    if data:
                        print(data, end='', flush=True)
                
                # 检查用户输入
                try:
                    import msvcrt
                    if msvcrt.kbhit():
                        char = msvcrt.getwch()
                        if char == '\r':
                            print()  # 换行
                            user_input = input_buffer
                            input_buffer = ''
                            
                            if user_input.strip().lower() in ['exit', 'quit']:
                                channel.send("exit\n")
                                break
                            channel.send(user_input + '\n')
                        else:
                            print(char, end='', flush=True)
                            input_buffer += char
                except ImportError:
                    # 非Windows系统
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"\n❌ 交互式Shell错误: {e}")
    
    def _read_channel(self, channel, timeout: float = 0.1) -> str:
        """从通道读取数据"""
        try:
            import select
        except ImportError:
            # Windows系统
            output = ""
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if channel.recv_ready():
                    data = channel.recv(1024).decode('utf-8', errors='ignore')
                    if data:
                        output += data
                        start_time = time.time()
                else:
                    time.sleep(0.01)
            
            return output
        
        # Unix系统
        output = ""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if select.select([channel], [], [], 0)[0]:
                data = channel.recv(1024).decode('utf-8', errors='ignore')
                if data:
                    output += data
                    start_time = time.time()
            else:
                time.sleep(0.01)
        
        return output
    
    def run_script(self, 
                  commands: List[str],
                  delay: float = 0.5,
                  show_output: bool = True) -> List[Dict]:
        """
        运行脚本（一系列命令）
        """
        print(f"📜 运行脚本 ({len(commands)} 个命令)")
        return self.execute_multiple_commands(commands, show_output=show_output)
    
    # ==================== 实用工具函数 ====================
    
    def _format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def get_server_info(self) -> Dict:
        """获取服务器信息"""
        info = {}
        
        # 执行多个系统命令
        commands = [
            ("hostname", "hostname"),
            ("kernel", "uname -a"),
            ("cpu", "lscpu | grep 'Model name' | cut -d':' -f2 | xargs"),
            ("memory", "free -h | grep 'Mem:' | awk '{print $2}'"),
            ("disk", "df -h / | tail -1 | awk '{print $4}'"),
            ("uptime", "uptime -p"),
            ("users", "who | wc -l")
        ]
        
        for name, cmd in commands:
            result = self.execute_command(cmd, show_output=False, capture_output=True)
            if result['success'] and 'stdout' in result:
                info[name] = result['stdout'].strip()
        
        return info
    
    def create_remote_directory(self, remote_dir: str) -> bool:
        """创建远程目录"""
        return self.execute_command(f"mkdir -p {remote_dir}")['success']
    
    def file_exists(self, remote_path: str) -> bool:
        """检查远程文件是否存在"""
        result = self.execute_command(f"test -f {remote_path} && echo 'exists'", 
                                     show_output=False, capture_output=True)
        return 'exists' in result.get('stdout', '')
    
    def directory_exists(self, remote_path: str) -> bool:
        """检查远程目录是否存在"""
        result = self.execute_command(f"test -d {remote_path} && echo 'exists'", 
                                     show_output=False, capture_output=True)
        return 'exists' in result.get('stdout', '')
    
    def get_file_content(self, remote_path: str) -> str:
        """获取远程文件内容"""
        result = self.execute_command(f"cat {remote_path}", 
                                     show_output=False, capture_output=True)
        return result.get('stdout', '') if result.get('success') else ''
    
    def put_file_content(self, remote_path: str, content: str) -> bool:
        """将内容写入远程文件"""
        # 转义内容中的特殊字符
        escaped_content = content.replace("'", "'\"'\"'")
        result = self.execute_command(f"echo '{escaped_content}' > {remote_path}")
        return result['success']
    
    def tail_file(self, remote_path: str, lines: int = 10, follow: bool = False) -> None:
        """查看文件尾部（类似tail命令）"""
        cmd = f"tail -n {lines} {remote_path}"
        if follow:
            cmd = f"tail -f {remote_path}"
        
        self.execute_command(cmd, show_output=True, capture_output=False)
    
    def watch_process(self, process_name: str, interval: float = 2.0, count: int = 10) -> None:
        """监控进程"""
        for i in range(count):
            print(f"\n📊 第 {i+1}/{count} 次监控")
            self.execute_command(f"ps aux | grep -E '{process_name}' | grep -v grep")
            time.sleep(interval)
    
    # ==================== 上下文管理器 ====================
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

# 导入select（Windows上不可用，但我们可以处理）
try:
    import select
except ImportError:
    pass

# ==================== 快捷函数 ====================

def create_ssh_client(host: str = "region-9.autodl.pro",
                     port: int = 49476,
                     username: str = "root",
                     password: str = "sCSRzjrEPVeT") -> UnifiedSSHClient:
    """创建并连接SSH客户端"""
    client = UnifiedSSHClient(host, port, username, password)
    client.connect()
    return client

def quick_upload(local_path: str, 
                remote_dir: str = "/root/autodl_data",
                **kwargs) -> Dict:
    """快速上传文件"""
    with UnifiedSSHClient() as client:
        filename = os.path.basename(local_path)
        remote_path = os.path.join(remote_dir, filename)
        return client.upload_file(local_path, remote_path, **kwargs)

def quick_download(remote_path: str,
                  local_dir: str = ".",
                  **kwargs) -> Dict:
    """快速下载文件"""
    with UnifiedSSHClient() as client:
        filename = os.path.basename(remote_path)
        local_path = os.path.join(local_dir, filename)
        return client.download_file(remote_path, local_path, **kwargs)

def quick_command(command: str, **kwargs) -> str:
    """快速执行命令并返回输出"""
    with UnifiedSSHClient() as client:
        result = client.execute_command(command, **kwargs)
        if result.get('success'):
            return result.get('stdout', '').strip()
        return result.get('error', '')

# ==================== 使用示例 ====================

if __name__ == "__main__":
    """
    使用示例
    """
    import sys
    
    print("=" * 60)
    print("🤖 统一SSH客户端工具")
    print("=" * 60)
    
    # 命令行参数
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "upload" and len(sys.argv) > 2:
            result = quick_upload(sys.argv[2])
            print("✅ 上传成功" if result.get('success') else "❌ 上传失败")
        
        elif mode == "download" and len(sys.argv) > 2:
            result = quick_download(sys.argv[2])
            print("✅ 下载成功" if result.get('success') else "❌ 下载失败")
        
        elif mode == "cmd" and len(sys.argv) > 2:
            output = quick_command(" ".join(sys.argv[2:]))
            print(output)
        
        elif mode == "shell":
            with UnifiedSSHClient() as client:
                client.interactive_shell()
        
        else:
            print("用法:")
            print("  python ssh_client.py upload <本地文件>")
            print("  python ssh_client.py download <远程文件>")
            print("  python ssh_client.py cmd <命令>")
            print("  python ssh_client.py shell")
        
        sys.exit(0)
    
    # 交互式菜单
    print("\n选择操作:")
    print("1. 🔌 连接并显示服务器信息")
    print("2. 📤 上传文件")
    print("3. 📥 下载文件")
    print("4. 💻 执行命令")
    print("5. 🖥️  交互式Shell")
    print("6. 📁 列出远程目录")
    print("7. 📊 系统监控")
    print("8. 🧪 运行示例脚本")
    
    choice = input("\n请选择 (1-8): ").strip()
    
    with UnifiedSSHClient() as client:
        if choice == "1":
            # 显示服务器信息
            print("\n🖥️ 服务器信息:")
            info = client.get_server_info()
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        elif choice == "2":
            # 上传文件
            local_path = input("请输入本地文件路径: ").strip()
            remote_dir = input("远程目录 (默认: /root/autodl_data): ").strip() or "/root/autodl_data"
            
            filename = os.path.basename(local_path)
            remote_path = os.path.join(remote_dir, filename)
            
            # 进度显示
            def show_progress(transferred, total):
                percent = (transferred / total) * 100
                bar = '█' * int(percent/2) + '░' * (50 - int(percent/2))
                print(f"\r  [{bar}] {percent:.1f}%", end='')
            
            result = client.upload_file(local_path, remote_path, callback=show_progress, overwrite=True)
            if result.get('success'):
                print(f"\n✅ 上传成功")
            else:
                print(f"\n❌ 上传失败: {result.get('error')}")
        
        elif choice == "3":
            # 下载文件
            remote_path = input("请输入远程文件路径: ").strip()
            local_dir = input("本地保存目录 (默认: 当前目录): ").strip() or "."
            
            result = client.download_file(remote_path, os.path.join(local_dir, os.path.basename(remote_path)))
            if result.get('success'):
                print("✅ 下载成功")
            else:
                print(f"❌ 下载失败: {result.get('error')}")
        
        elif choice == "4":
            # 执行命令
            command = input("请输入要执行的命令: ").strip()
            result = client.execute_command(command)
            if result.get('success'):
                print(f"✅ 命令执行成功 (退出码: {result.get('exit_code')})")
            else:
                print(f"❌ 命令执行失败: {result.get('error')}")
        
        elif choice == "5":
            # 交互式Shell
            client.interactive_shell()
        
        elif choice == "6":
            # 列出目录
            remote_path = input("要列出的目录 (默认: 当前目录): ").strip() or "."
            items = client.list_directory(remote_path)
            
            if items:
                print(f"\n📁 目录内容: {remote_path}")
                print("-" * 60)
                for item in items:
                    icon = "📁" if item['type'] == 'directory' else "📄"
                    print(f"{icon} {item['permissions']} {item['size']:>8} {item['modified']} {item['name']}")
            else:
                print("❌ 无法列出目录或目录为空")
        
        elif choice == "7":
            # 系统监控
            print("📊 系统监控 (Ctrl+C退出)")
            try:
                while True:
                    print("\n" + "="*50)
                    print(time.strftime("%Y-%m-%d %H:%M:%S"))
                    print("="*50)
                    
                    commands = [
                        "uptime",
                        "free -h | head -2",
                        "df -h / | tail -1",
                        "top -bn1 | head -5"
                    ]
                    
                    for cmd in commands:
                        print(f"\n{cmd}:")
                        client.execute_command(cmd, show_output=True, capture_output=False)
                    
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                print("\n👋 停止监控")
        
        elif choice == "8":
            # 运行示例脚本
            print("🧪 运行示例脚本")
            
            script = [
                "echo '=== 系统信息 ==='",
                "uname -a",
                "echo ''",
                "echo '=== 工作目录 ==='",
                "pwd",
                "ls -la",
                "echo ''",
                "echo '=== 磁盘空间 ==='",
                "df -h",
                "echo ''",
                "echo '=== 内存使用 ==='",
                "free -h",
                "echo ''",
                "echo '=== 示例完成 ==='"
            ]
            
            client.run_script(script)
        
        else:
            print("❌ 无效选择")
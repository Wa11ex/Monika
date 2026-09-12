import asyncio
import pyvts
import os
import sys
from utils.config_loader import load_config, get_config


async def generate_vts_token():
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)
    
    load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
    
    plugin_name = get_config("vts.plugin_name", "Monika_Core")
    developer = get_config("vts.developer", "YourName")
    token_path = get_config("vts.token_path", "./vts_token.txt")
    
    if not os.path.isabs(token_path):
        token_path = os.path.join(PROJECT_ROOT, token_path)
    
    print("=" * 50)
    print("    VTube Studio Token 生成工具")
    print("=" * 50)
    print(f"插件名: {plugin_name}")
    print(f"开发者: {developer}")
    print(f"Token保存路径: {token_path}")
    print("=" * 50)
    
    if os.path.exists(token_path):
        print(f"\n✓ Token 文件已存在: {token_path}")
        response = input("\n是否重新生成 Token？(y/n): ").strip().lower()
        if response != 'y':
            print("取消生成，退出")
            return
        else:
            os.remove(token_path)
            print("已删除旧 Token")
    
    print("\n正在连接 VTube Studio...")
    print("请在 VTube Studio 中点击允许授权！\n")

    vts = pyvts.vts(plugin_info={
        "plugin_name": plugin_name,
        "developer": developer,
        "authentication_token_path": token_path
    })
    
    try:
        await vts.connect()
        print("✓ 已连接到 VTube Studio")
        
        print("请求认证 Token...")
        await vts.request_authenticate_token()
        print("✓ Token 授权请求已发送，请在 VTube Studio 中点击允许")
        
        for i in range(30):
            try:
                if await vts.request_authenticate():
                    print("\n✓ 认证成功！")
                    print(f"✓ Token 已保存到: {token_path}")
                    print("\n你现在可以运行 main.py 了！")
                    await vts.close()
                    return
            except:
                pass
            
            await asyncio.sleep(1)
            print(".", end="", flush=True)
        
        print("\n✗ 认证超时，请检查 VTube Studio 是否已启动并允许插件连接")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        print("\n排查步骤:")
        print("1. VTube Studio 已启动？")
        print("2. 设置 -> API 中是否启用了插件？")
        print("3. VTube Studio 是否在弹窗要求授权？")
    finally:
        await vts.close()


def main():
    try:
        asyncio.run(generate_vts_token())
    except KeyboardInterrupt:
        print("\n\n取消了 Token 生成")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
"""
Helper utilities for TensorBoard compatibility and troubleshooting.
"""
import os
import sys
import logging
import subprocess
import pkg_resources

def check_tensorboard_compatibility():
    """Check if TensorBoard is properly installed with compatible dependencies."""
    try:
        import tensorboard
        tb_version = pkg_resources.get_distribution("tensorboard").version
        
        # Check for protobuf version - a common source of compatibility issues
        protobuf_version = None
        try:
            protobuf_version = pkg_resources.get_distribution("protobuf").version
        except:
            pass
        
        print(f"TensorBoard version: {tb_version}")
        print(f"Protobuf version: {protobuf_version or 'Unknown'}")
        
        # Simple compatibility check for common issues
        if tb_version.startswith("2.") and protobuf_version:
            if protobuf_version.startswith("4."):
                print("Warning: TensorBoard 2.x may have compatibility issues with protobuf 4.x")
                print("Consider downgrading protobuf: pip install protobuf==3.20.3")
        
        return True
    except ImportError:
        print("TensorBoard is not installed.")
        return False
    except Exception as e:
        print(f"Error checking TensorBoard: {str(e)}")
        return False

def install_compatible_dependencies():
    """Install compatible versions of TensorBoard dependencies."""
    try:
        print("Installing TensorBoard with compatible dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard==2.11.0", "protobuf==3.20.3"])
        print("Installation completed. Please restart your application.")
        return True
    except Exception as e:
        print(f"Installation failed: {str(e)}")
        return False

def launch_tensorboard(logdir="runs", port=6006):
    """Launch TensorBoard with the specified logdir."""
    try:
        cmd = [sys.executable, "-m", "tensorboard.main", f"--logdir={logdir}", f"--port={port}"]
        print(f"Launching TensorBoard: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        print(f"TensorBoard running at http://localhost:{port}/")
        return process
    except Exception as e:
        print(f"Failed to launch TensorBoard: {str(e)}")
        return None

if __name__ == "__main__":
    # Simple CLI for the helper
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorBoard Helper Utilities")
    parser.add_argument("--check", action="store_true", help="Check TensorBoard compatibility")
    parser.add_argument("--fix", action="store_true", help="Install compatible dependencies")
    parser.add_argument("--launch", action="store_true", help="Launch TensorBoard")
    parser.add_argument("--logdir", default="runs", help="Log directory for TensorBoard")
    parser.add_argument("--port", type=int, default=6006, help="Port for TensorBoard server")
    
    args = parser.parse_args()
    
    if args.check:
        check_tensorboard_compatibility()
    elif args.fix:
        install_compatible_dependencies()
    elif args.launch:
        process = launch_tensorboard(args.logdir, args.port)
        if process:
            try:
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
                print("TensorBoard stopped.")
    else:
        parser.print_help()

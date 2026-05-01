{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.numpy
    pkgs.python3Packages.networkx
    pkgs.python3Packages.torch
    pkgs.gcc
    pkgs.zlib
  ];

  shellHook = ''
    # Create venv if it doesn't exist
    if [ ! -d venv ]; then
      python -m venv venv
      source venv/bin/activate
      pip install torch_geometric --break-system-packages 2>/dev/null || \
      pip install torch_geometric
    fi
    source venv/bin/activate
    export PYTHONPATH="$PWD:$PYTHONPATH"
  '';
}

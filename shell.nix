{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs.python3Packages; [
    pkgs.python3
    numpy
    networkx
    torch
    torchvision
    matplotlib
    seaborn
    pandas
    tqdm
    pytest
    pillow
    pkgs.gcc
    pkgs.zlib
    pkgs.git
  ];

  shellHook = ''
    # Create venv if it doesn't exist and install torch_geometric
    if [ ! -d venv ]; then
      python -m venv venv
    fi
    source venv/bin/activate
    pip install torch_geometric 2>/dev/null || echo "torch_geometric may already be installed"

    export PYTHONPATH="$PWD:$PYTHONPATH"
    echo "Nix shell ready. PYTHONPATH=$PYTHONPATH"
  '';
}

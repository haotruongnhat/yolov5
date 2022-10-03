# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

a = Analysis(
    ['demthep_detect.py'],
    pathex=[],
    binaries=[('d:\\miniconda3\\envs\\yolox-build\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime_providers_shared.dll', './onnxruntime/capi/'),
                ('d:\\miniconda3\\envs\\yolox-build\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime_providers_cuda.dll', './onnxruntime/capi/'),
                ('d:\\miniconda3\\envs\\yolox-build\\Lib\\site-packages\\onnxruntime\\capi\\onnxruntime_providers_tensorrt.dll', './onnxruntime/capi/')],
    datas=[],
    hiddenimports=['sklearn.utils._typedefs',
                    'sklearn.utils._cython_blas',
                    'sklearn.neighbors.typedefs',
                    'sklearn.neighbors.quad_tree',
                    'sklearn.neighbors._partition_nodes',
                    'sklearn.tree._utils',
                    'sklearn.neighbors._typedefs',
                    'sklearn.utils._weight_vector',
                    'sklearn.neighbors._quad_tree'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='demthep_detect_test_udp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='demthep.ico'
)

# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['deg_volcano.py'],
    pathex=[],
    binaries=[],
    datas=[('29081.csv', '.'), ('115828.csv', '.'), ('135092.csv', '.'), ('99248.csv', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DEGVolcano',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
app = BUNDLE(
    exe,
    name='DEGVolcano.app',
    icon=None,
    bundle_identifier=None,
)

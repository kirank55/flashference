"""Read one tensor from a GGUF (bf16/f16/f32) as float32 numpy. usage: from gguf_read import read_tensor"""
import struct, numpy as np
def _rd(f, fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))
def _rstr(f): (n,) = _rd(f, "<Q"); return f.read(n).decode("utf8", "replace")
_SZ = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
def _rval(f, t):
    if t in _SZ: return _rd(f, "<" + _SZ[t])[0]
    if t == 8: return _rstr(f)
    if t == 9:
        (et,) = _rd(f, "<I"); (n,) = _rd(f, "<Q")
        if et in _SZ and et != 7: f.seek(struct.calcsize(_SZ[et]) * n, 1); return None
        for _ in range(n): _rval(f, et)
def read_tensor(path, name):
    f = open(path, "rb"); f.read(4); _rd(f, "<I"); nt, = _rd(f, "<Q"); nkv, = _rd(f, "<Q"); align = 32
    for _ in range(nkv):
        k = _rstr(f); t, = _rd(f, "<I"); v = _rval(f, t)
        if k == "general.alignment": align = v
    info = {}
    for _ in range(nt):
        n = _rstr(f); nd, = _rd(f, "<I"); dims = _rd(f, "<%dQ" % nd); ty, = _rd(f, "<I"); off, = _rd(f, "<Q"); info[n] = (dims, ty, off)
    start = (f.tell() + align - 1) // align * align
    dims, ty, off = info[name]; n = int(np.prod(dims))
    f.seek(start + off)
    if ty == 30:
        a = np.frombuffer(f.read(n * 2), dtype=np.uint16).astype(np.uint32) << 16
        x = a.view(np.float32)
    elif ty == 1: x = np.frombuffer(f.read(n * 2), dtype=np.float16).astype(np.float32)
    elif ty == 0: x = np.frombuffer(f.read(n * 4), dtype=np.float32)
    else: raise ValueError("type %d" % ty)
    return x.reshape(dims[::-1])

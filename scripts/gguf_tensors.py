import struct,sys
def rd(f,fmt): 
    s=struct.calcsize(fmt); return struct.unpack(fmt,f.read(s))
def rstr(f): (n,)=rd(f,"<Q"); return f.read(n).decode("utf8","replace")
SZ={0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
def rval(f,t):
    if t in SZ: return rd(f,"<"+SZ[t])[0]
    if t==8: return rstr(f)
    if t==9:
        (et,)=rd(f,"<I"); (n,)=rd(f,"<Q")
        if et in SZ and et!=7:
            f.seek(struct.calcsize(SZ[et])*n,1); return "arr"
        for _ in range(n): rval(f,et)
        return "arr"
for fn in sys.argv[1:]:
    f=open(fn,"rb"); f.read(4); ver,=rd(f,"<I"); nt,=rd(f,"<Q"); nkv,=rd(f,"<Q")
    for _ in range(nkv):
        k=rstr(f); t,=rd(f,"<I"); v=rval(f,t)
        if k.startswith("qwen35") or k=="general.file_type": print(" ",k,v)
    T=[]
    for _ in range(nt):
        n=rstr(f); nd,=rd(f,"<I"); dims=rd(f,"<%dQ"%nd); ty,=rd(f,"<I"); off,=rd(f,"<Q"); T.append((n,dims,ty))
    print(fn,len(T),"tensors")
    for n,d,ty in T:
        if "blk.24" in n or "nextn" in n or "mtp" in n or n.startswith("token_embd") or n.startswith("output"): print("  ",n,d,ty)

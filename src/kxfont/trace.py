"""Minimal Potrace public C API binding. Output uses source-pixel coordinates."""
import ctypes as C,ctypes.util,os
import numpy as np
class Point(C.Structure): _fields_=[('x',C.c_double),('y',C.c_double)]
class Curve(C.Structure): _fields_=[('n',C.c_int),('tag',C.POINTER(C.c_int)),('c',C.POINTER(Point*3))]
class Path(C.Structure): pass
Path._fields_=[('area',C.c_int),('sign',C.c_int),('curve',Curve),('next',C.POINTER(Path)),('childlist',C.POINTER(Path)),('sibling',C.POINTER(Path)),('priv',C.c_void_p)]
class State(C.Structure): _fields_=[('status',C.c_int),('plist',C.POINTER(Path)),('priv',C.c_void_p)]
class Progress(C.Structure): _fields_=[('callback',C.c_void_p),('data',C.c_void_p),('min',C.c_double),('max',C.c_double),('epsilon',C.c_double)]
class Param(C.Structure): _fields_=[('turdsize',C.c_int),('turnpolicy',C.c_int),('alphamax',C.c_double),('opticurve',C.c_int),('opttolerance',C.c_double),('progress',Progress)]
class Bitmap(C.Structure): _fields_=[('w',C.c_int),('h',C.c_int),('dy',C.c_int),('map',C.POINTER(C.c_ulong))]
_lib=None
def library():
 global _lib
 if _lib is None:
  path=os.environ.get('POTRACE_LIBRARY') or ctypes.util.find_library('potrace')
  if not path and os.name=='nt':path='E:/FontForgeBuilds/bin/libpotrace-0.dll'
  if not path:raise RuntimeError('Install libpotrace or set POTRACE_LIBRARY')
  _lib=C.CDLL(path);_lib.potrace_param_default.restype=C.POINTER(Param)
  _lib.potrace_trace.argtypes=[C.POINTER(Param),C.POINTER(Bitmap)];_lib.potrace_trace.restype=C.POINTER(State)
  _lib.potrace_param_free.argtypes=[C.POINTER(Param)];_lib.potrace_state_free.argtypes=[C.POINTER(State)]
  _lib.potrace_version.restype=C.c_char_p
 return _lib
def trace(mask,tolerance=.20,alpha=1.0):
 lib=library();h,w=mask.shape;bits=C.sizeof(C.c_ulong)*8;dy=(w+bits-1)//bits
 a=np.zeros((h,dy*bits),np.uint8);a[:,:w]=mask
 packed=np.packbits(a,axis=1).reshape(h,dy,bits//8)
 dtype=np.uint32 if bits==32 else np.uint64
 words=np.zeros((h,dy),dtype=dtype)
 for j in range(bits//8):words|=packed[:,:,j].astype(dtype)<<((bits//8-1-j)*8)
 bm=Bitmap(w,h,dy,words.ctypes.data_as(C.POINTER(C.c_ulong)));p=lib.potrace_param_default()
 p.contents.turdsize=0;p.contents.alphamax=alpha;p.contents.opttolerance=tolerance
 st=lib.potrace_trace(p,C.byref(bm))
 try:
  if not st or st.contents.status!=0:raise RuntimeError('Potrace failed')
  out=[];ptr=st.contents.plist
  while ptr:
   q=ptr.contents;cv=q.curve;segments=[]
   for i in range(cv.n):segments.append((cv.tag[i],[(cv.c[i][j].x,cv.c[i][j].y) for j in range(3)]))
   out.append((q.sign,q.area,segments));ptr=q.next
  return out
 finally:
  if st:lib.potrace_state_free(st)
  lib.potrace_param_free(p)
def draw(paths,pen,transform=lambda x,y:(x,y)):
 for sign,area,segs in paths:
  pen.moveTo(transform(*segs[-1][1][2]))
  for tag,p in segs:
   if tag==2:pen.lineTo(transform(*p[1]));pen.lineTo(transform(*p[2]))
   else:pen.curveTo(*[transform(*v) for v in p])
  pen.closePath()

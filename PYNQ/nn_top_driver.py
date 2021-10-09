class nn_top_Driver:
    def __init__(self, bit_path=None):
        self.overlay = Overlay(bit_path)
        self.IP = self. overlay.nn_top_0
        self.reg = self.IP.register_map
        self.base_addr = self.IP.mmio.base_addr
        #Input and Output buffers
        self.input_buffer = None
        self.output_buffer = None
        
#         #timer
#         self.timer = self.overlay.axi_timer_0
#         self.timer_reg = self.timer.register_map
    
    
#     def start_timer(self):
        
    
    
    def run(self):
        self.reg.CTRL.AP_START = 1
        while (not self.reg.CTRL.AP_DONE):
            pass
        
        
#   in_V_1 = Register(in_V=0),
#   in_V_2 = Register(in_V=0),
#   out_V_1 = Register(out_V=0),
#   out_V_2 = Register(out_V=0),
#   doInit = Register(doInit=0, RESERVED=0),
#   targetmem = Register(targetmem=0),
#   target_ch = Register(target_ch=0),
#   target_row = Register(target_row=0),
#   target_col = Register(target_col=0),
#   val_V_1 = Register(val_V=0),
#   val_V_2 = Register(val_V=0),
#   numReps = Register(numReps=0)

    def init_so(self):
        dllname = "layer_loader.so"
        LIB_PATH = "/home/xilinx/jupyter_notebooks/Abdullah/lib"
        _libraries[dllname] = _ffi.dlopen(os.path.join( LIB_PATH, dllname))
        self.interface = _libraries[dllname]
        self.interface.load_layer(self.base_addr)
        

    def initialize(self, file, targetmem, CH, ROW, COL, isbias):
        
        if(isbias):
            array = np.loadtxt(file, dtype=ctypes.c_longlong, delimiter=',', usecols=[0])
        else:
            array = np.loadtxt(file, dtype=ctypes.c_uint32, delimiter=',',
                           usecols=[0],
                           converters={_:lambda s: int(s, 16) for _ in range(CH*ROW*COL)})

        array = array.reshape([CH, ROW, COL])
        print('Writing parameters...')
#         print(array)
        for ch in range(CH):
            for row in range(ROW):
                for col in range(COL):
#                     if (ch, row,  col) != (0, 0, 2):
#                         continue
                    print('ch:{}, row:{}, col:{}'.format(str(ch), str(row), str(col)))
                    self.reg.doInit = 1
                    self.reg.targetmem = targetmem
                    self.reg.target_ch = ch
                    self.reg.target_row = row
                    self.reg.target_col = col
                    self.reg.numReps = 1
                    
                    if(isbias):
                        self.reg.val_V_1 = int(array[ch,row,col]) & (0xffffffff)
                        self.reg.val_V_2 = int(array[ch,row,col]) >> 32 & (0xffffffff)
                    else:    
                        self.reg.val_V_1 = (array[ch,row,col]) & (0xffffffff)
                        self.reg.val_V_2 = (array[ch,row,col]) >> 32 & (0xffffffff)
                    
                    # garbage vals
                    self.reg.in_V_1 = 0
                    self.reg.in_V_2 = 0
                    self.reg.out_V_1 = 0
                    self.reg.out_V_2 = 0
                    # Run IP
                    self.run()
                    self.reg.doInit = 0
    
    def set_io(self, file):
        in_shape = (28*28)//8 #due to 64 bit memory
        out_shape = 490
        
        image = np.loadtxt(file,
                           dtype=np.uint64, delimiter=',', usecols=[0],
                           converters={_:lambda s: int(s, 16) for _ in range(in_shape)})
        
        self.input_buffer = allocate(shape=in_shape, dtype=np.uint64)
        self.output_buffer = allocate(shape=out_shape, dtype=np.uint16)
        
        np.copyto(self.input_buffer, image)
        np.copyto(self.output_buffer, np.zeros(out_shape, dtype=np.uint16))
        
        self.reg.in_V_1 = self.input_buffer.physical_address & 0xffffffff
        self.reg.in_V_2 = (self.input_buffer.physical_address >> 32) & 0xffffffff
        self.reg.out_V_1 = (self.output_buffer.physical_address) & 0xffffffff
        self.reg.out_V_2 = (self.output_buffer.physical_address >> 32) & 0xffffffff
        self.reg.numReps = 1
        
        self.reg.doInit = 0
        self.reg.targetmem = 0
        self.reg.target_ch = 0
        self.reg.target_row = 0
        self.reg.target_col = 0
        self.reg.val_V_1 = 0
        self.reg.val_V_2 = 0
        print("Image Set. Ready to run")


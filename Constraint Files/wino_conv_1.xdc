set_property PREG 1 [all_dsps]
set_property MREG 1 [filter [all_dsps] {USE_MULT == MULTIPLY}]
#set_property AREG 1 [filter [all_dsps] {ACASCREG == 1}]
#set_property BREG 2 [all_dsps]
#set_property SCOPED_TO_REF wino_conv_1 [
#    set_property PREG 1 [all_dsps],
#    set_property MREG 1 [filter [all_dsps] {USE_MULT == MULTIPLY}]]
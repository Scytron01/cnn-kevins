transcript on
if {[file exists work_cnn]} { vdel -lib work_cnn -all }
vlib work_cnn
vlog -2001 -work work_cnn verilog/cnn_mac_unit.v
vlog -2001 -work work_cnn verilog/cnn_top.v
vlog -2001 -work work_cnn verilog/tb_cnn_top.v
vsim -voptargs=+acc work_cnn.tb_cnn_top
run -all
quit -f

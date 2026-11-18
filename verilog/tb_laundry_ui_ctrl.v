`timescale 1ns/1ps
module tb_laundry_ui_ctrl;
    reg clk=0,reset_n=0,avs_read=0,avs_write=0;
    reg [3:0] key_n=4'hF; reg [9:0] switches=0;
    reg [1:0] avs_address=0; reg [31:0] avs_writedata=0;
    wire [31:0] avs_readdata; wire avs_waitrequest;
    laundry_ui_ctrl #(.DEBOUNCE_CYCLES(4)) dut(
        .clk(clk),.reset_n(reset_n),.key_n(key_n),.switches(switches),
        .avs_address(avs_address),.avs_read(avs_read),.avs_readdata(avs_readdata),
        .avs_write(avs_write),.avs_writedata(avs_writedata),.avs_waitrequest(avs_waitrequest));
    always #5 clk=~clk;
    initial begin
        repeat(3) @(negedge clk); reset_n=1; key_n[0]=0; repeat(12) @(negedge clk);
        avs_address=1; avs_read=1; #1;
        if(avs_readdata[0]!==1) $fatal(1,"KEY0 event was not latched");
        @(negedge clk); avs_read=0; avs_write=1; avs_writedata=1;
        @(negedge clk); avs_write=0; avs_read=1; #1;
        if(avs_readdata[0]!==0) $fatal(1,"KEY0 event did not clear");
        switches=10'b10; avs_address=2; #1;
        if(avs_readdata[1:0]!==2'b10) $fatal(1,"SW[1:0] mismatch");
        $display("PASS: LAUNDRY_UI_CTRL"); $finish;
    end
endmodule

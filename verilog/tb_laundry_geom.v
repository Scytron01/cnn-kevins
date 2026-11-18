`timescale 1ns/1ps
module tb_laundry_geom;
  reg clk=0, reset_n=0;
  reg sink_valid=0,sink_data=0,sink_sop=0,sink_eop=0;
  wire sink_ready,irq;
  reg [2:0] csr_address=0;
  reg csr_read=0;
  wire [31:0] csr_readdata;
  integer r,c,errors=0;
  reg px;
  laundry_geom_ip dut(.clk(clk),.reset_n(reset_n),.sink_valid(sink_valid),
    .sink_data(sink_data),.sink_sop(sink_sop),.sink_eop(sink_eop),
    .sink_ready(sink_ready),.csr_address(csr_address),.csr_read(csr_read),
    .csr_readdata(csr_readdata),.irq(irq));
  always #10 clk=~clk;
  task read_reg(input [2:0] a);
    begin @(negedge clk); csr_address=a; csr_read=1; #1; end
  endtask
  initial begin
    #60 reset_n=1;
    for(r=0;r<64;r=r+1) begin
      for(c=0;c<64;c=c+1) begin
        px=0;
        if(r>=8 && r<=30 && c>=20 && c<=44) px=1;
        else if(r>=31 && r<=58 && ((c>=20 && c<=29)||(c>=35 && c<=44))) px=1;
        @(negedge clk); sink_valid=1; sink_data=px;
        sink_sop=(r==0 && c==0); sink_eop=(r==63 && c==63);
        if(!sink_ready) begin $display("FAIL: sink not ready"); errors=errors+1; end
      end
    end
    @(negedge clk); sink_valid=0; sink_sop=0; sink_eop=0;
    wait(irq==1); #2;
    read_reg(1); $display("AREA=%0d",csr_readdata[12:0]);
    if(csr_readdata[12:0]!=1135) errors=errors+1;
    read_reg(2); $display("BBOX=%0d,%0d,%0d,%0d",csr_readdata[7:0],csr_readdata[15:8],csr_readdata[23:16],csr_readdata[31:24]);
    if(csr_readdata[7:0]!=20 || csr_readdata[15:8]!=44 || csr_readdata[23:16]!=8 || csr_readdata[31:24]!=58) errors=errors+1;
    read_reg(3); $display("TWO_RUN_ROWS=%0d",csr_readdata[7:0]);
    if(csr_readdata[7:0]==0) errors=errors+1;
    if(errors==0) $display("PASS: LAUNDRY GEOMETRY"); else $display("FAIL: %0d",errors);
    #20 $finish;
  end
endmodule

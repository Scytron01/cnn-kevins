`timescale 1ns/1ps
module tb_cnn_avalon_slave;
    reg clk=0,reset_n=0,avs_read=0,avs_write=0;
    reg [7:0] avs_address=0; reg [31:0] avs_writedata=0;
    wire [31:0] avs_readdata; wire avs_waitrequest,irq;
    integer i,watchdog; reg [31:0] status,result,cycles;
    cnn_avalon_slave dut(.clk(clk),.reset_n(reset_n),.avs_address(avs_address),
        .avs_read(avs_read),.avs_readdata(avs_readdata),.avs_write(avs_write),
        .avs_writedata(avs_writedata),.avs_waitrequest(avs_waitrequest),.irq(irq));
    always #10 clk=~clk;
    task write_word; input [7:0] address; input [31:0] value;
        begin @(negedge clk); avs_address=address; avs_writedata=value; avs_write=1;
        @(negedge clk); avs_write=0; end
    endtask
    task read_word; input [7:0] address; output [31:0] value;
        begin @(negedge clk); avs_address=address; avs_read=1; #1; value=avs_readdata;
        @(negedge clk); avs_read=0; end
    endtask
    initial begin
        repeat(5) @(negedge clk); reset_n=1;
        for(i=0;i<128;i=i+1) write_word(8'h40+i,0);
        write_word(8'h02,0); write_word(8'h00,1);
        watchdog=0; status=0;
        while(status[1]!==1 && watchdog<3500000) begin read_word(8'h01,status); watchdog=watchdog+1; end
        if(status[1]!==1) $fatal(1,"CNN wrapper timeout after %0d polls",watchdog);
        read_word(8'h03,result); read_word(8'h04,cycles);
        if(result[2:0]!==0) $fatal(1,"Zero ROM tie-break should be class 0");
        if(cycles==0) $fatal(1,"Cycle counter did not increment");
        if(irq!==1) $fatal(1,"DONE interrupt did not assert");
        $display("PASS: CNN_AVALON_SLAVE class=%0d cycles=%0d",result[2:0],cycles); $finish;
    end
endmodule

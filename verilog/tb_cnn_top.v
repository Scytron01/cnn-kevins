`timescale 1ns/1ps
module tb_cnn_top;
    reg clk=0,reset_n=0,start=0,in_valid=0,in_pixel=0;
    reg [31:0] geometry_q7=0;
    wire in_ready,result_valid,busy;
    wire [2:0] result_class;
    integer i,watchdog;
    cnn_top dut(.clk(clk),.reset_n(reset_n),.start(start),.geometry_q7(geometry_q7),
        .in_valid(in_valid),.in_ready(in_ready),.in_pixel(in_pixel),
        .result_valid(result_valid),.result_class(result_class),.busy(busy));
    always #5 clk=~clk;
    initial begin
        watchdog=0;
        repeat(5) @(negedge clk);
        reset_n=1;
        @(negedge clk); start=1;
        @(negedge clk); start=0;
        for(i=0;i<4096;i=i+1) begin
            while(in_ready!==1'b1) @(negedge clk);
            in_valid=1;
            if(((i/64)>=8)&&((i/64)<28)&&((i%64)>=18)&&((i%64)<46))
                in_pixel=1;
            else if(((i/64)>=28)&&((i/64)<58)&&
                    ((((i%64)>=18)&&((i%64)<28))||(((i%64)>=36)&&((i%64)<46))))
                in_pixel=1;
            else in_pixel=0;
            @(negedge clk);
        end
        in_valid=0;
        while(result_valid!==1'b1) begin
            @(negedge clk);
            watchdog=watchdog+1;
            if(watchdog>3500000) begin
                $display("FAIL: watchdog expired while CNN was processing");
                $finish;
            end
        end
        if(result_class>3'd4) $display("FAIL: invalid class id %0d",result_class);
        else $display("PASS: CNN inference completed, predicted class id=%0d",result_class);
        @(negedge clk);
        $finish;
    end
endmodule

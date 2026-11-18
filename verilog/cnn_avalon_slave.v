// Avalon-MM wrapper around cnn_top.
// HPS writes 128 packed 32-bit words (4096 mask bits), then geometry Q7 and START.
// Platform Designer: addressUnits=WORDS. CPU byte offsets are register offsets * 4.
module cnn_avalon_slave (
    input wire clk, input wire reset_n,
    input wire [7:0] avs_address, input wire avs_read, output reg [31:0] avs_readdata,
    input wire avs_write, input wire [31:0] avs_writedata,
    output wire avs_waitrequest, output wire irq
);
    localparam [7:0] REG_CTRL=8'h00, REG_STATUS=8'h01, REG_GEOMETRY=8'h02,
                     REG_RESULT=8'h03, REG_CYCLES=8'h04, MASK_BASE=8'h40, MASK_LAST=8'hBF;
    reg [31:0] mask_words [0:127];
    reg [31:0] geometry_q7, cycle_counter;
    reg [2:0] result_class_reg;
    reg done_latched, start_pending, feed_active;
    reg [12:0] feed_count;
    reg cnn_start;
    wire cnn_in_ready, cnn_busy, cnn_result_valid;
    wire [2:0] cnn_result_class;
    wire cnn_in_valid=feed_active && cnn_in_ready;
    wire cnn_in_pixel=mask_words[feed_count[11:5]][feed_count[4:0]];
    wire system_busy=cnn_busy || start_pending || feed_active;
    assign avs_waitrequest=1'b0;
    assign irq=done_latched;

    cnn_top u_cnn_top(
        .clk(clk),.reset_n(reset_n),.start(cnn_start),.geometry_q7(geometry_q7),
        .in_valid(cnn_in_valid),.in_ready(cnn_in_ready),.in_pixel(cnn_in_pixel),
        .result_valid(cnn_result_valid),.result_class(cnn_result_class),.busy(cnn_busy)
    );

    // Mask RAM is deliberately not reset: HPS writes every word before START,
    // avoiding a reset-for-loop that would prevent efficient block-RAM inference.
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            geometry_q7<=0; cycle_counter<=0; result_class_reg<=0;
            done_latched<=0; start_pending<=0; feed_active<=0; feed_count<=0; cnn_start<=0;
        end else begin
            cnn_start<=0;
            if(avs_write && !system_busy) begin
                if(avs_address==REG_GEOMETRY)
                    geometry_q7<=avs_writedata;
                else if(avs_address>=MASK_BASE && avs_address<=MASK_LAST)
                    mask_words[avs_address-MASK_BASE]<=avs_writedata;
                else if(avs_address==REG_CTRL && avs_writedata[0]) begin
                    done_latched<=0; cycle_counter<=0; feed_count<=0;
                    start_pending<=1; cnn_start<=1;
                end
            end
            if(start_pending && cnn_in_ready) begin
                start_pending<=0; feed_active<=1; feed_count<=0;
            end
            if(cnn_in_valid) begin
                if(feed_count==13'd4095) feed_active<=0;
                else feed_count<=feed_count+1'b1;
            end
            if(system_busy) cycle_counter<=cycle_counter+1'b1;
            if(cnn_result_valid) begin
                result_class_reg<=cnn_result_class; done_latched<=1;
            end
        end
    end

    always @* begin
        avs_readdata=0;
        if(avs_read) begin
            case(avs_address)
                REG_CTRL: avs_readdata=0;
                REG_STATUS: avs_readdata={30'd0,done_latched,system_busy}; // bit1 DONE, bit0 BUSY
                REG_GEOMETRY: avs_readdata=geometry_q7;
                REG_RESULT: avs_readdata={29'd0,result_class_reg};
                REG_CYCLES: avs_readdata=cycle_counter;
                default: if(avs_address>=MASK_BASE && avs_address<=MASK_LAST)
                    avs_readdata=mask_words[avs_address-MASK_BASE];
            endcase
        end
    end
endmodule

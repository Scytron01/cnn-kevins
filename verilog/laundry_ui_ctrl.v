// Debounced active-low DE10-Standard KEYs and synchronized slide switches.
module laundry_ui_ctrl #(
    parameter integer DEBOUNCE_CYCLES=500000 // 10 ms at 50 MHz
)(
    input wire clk, input wire reset_n,
    input wire [3:0] key_n, input wire [9:0] switches,
    input wire [1:0] avs_address, input wire avs_read, output reg [31:0] avs_readdata,
    input wire avs_write, input wire [31:0] avs_writedata,
    output wire avs_waitrequest
);
    reg [3:0] key_meta,key_sync,pressed,key_events;
    reg [9:0] sw_meta,sw_sync;
    reg [31:0] debounce_count [0:3];
    wire [3:0] sampled_pressed=~key_sync;
    assign avs_waitrequest=1'b0;
    integer i;
    always @(posedge clk or negedge reset_n) begin
        if(!reset_n) begin
            key_meta<=4'hF; key_sync<=4'hF; pressed<=0; key_events<=0;
            sw_meta<=0; sw_sync<=0;
            for(i=0;i<4;i=i+1) debounce_count[i]<=0;
        end else begin
            key_meta<=key_n; key_sync<=key_meta; sw_meta<=switches; sw_sync<=sw_meta;
            for(i=0;i<4;i=i+1) begin
                if(sampled_pressed[i]==pressed[i]) debounce_count[i]<=0;
                else if(debounce_count[i]>=DEBOUNCE_CYCLES-1) begin
                    debounce_count[i]<=0; pressed[i]<=sampled_pressed[i];
                    if(sampled_pressed[i]) key_events[i]<=1'b1;
                end else debounce_count[i]<=debounce_count[i]+1'b1;
            end
            // Sticky event bits; software clears selected bits using write-one-to-clear.
            if(avs_write && avs_address==2'd1)
                key_events<=key_events & ~avs_writedata[3:0];
        end
    end
    always @* begin
        avs_readdata=0;
        if(avs_read) case(avs_address)
            2'd0: avs_readdata={24'd0,key_events,pressed};
            2'd1: avs_readdata={28'd0,key_events};
            2'd2: avs_readdata={22'd0,sw_sync};
            default: avs_readdata=0;
        endcase
    end
endmodule

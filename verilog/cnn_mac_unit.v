module cnn_mac_unit (
    input wire signed [7:0] activation,
    input wire signed [7:0] weight,
    output wire signed [15:0] product
);
    assign product = activation * weight;
endmodule

function check_filters(isize, kernel, stride, pad)
    println("input shape->", isize)
    while isize > 1
        float_out = (isize + 2*pad -kernel) / stride + 1 
        (float_out == isize) ? break : nothing
        isize = floor(float_out)
        println("output -> $(float_out) -> after floor -> $(isize) $((float_out != isize) ? " -> problem" : "")")
    end
end

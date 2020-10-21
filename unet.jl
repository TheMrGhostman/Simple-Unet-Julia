using Flux

struct UnetBlock
	model
	cat_pass::Bool # BOOL if concatenate input and output "cat(x, model(x))"" or not
	skip_pass
end

Flux.@functor UnetBlock

function (Unet::UnetBlock)(x; dims=3)
	if Unet.cat_pass
		return cat(Unet.skip_pass(x), Unet.model(x), dims=dims)
	else
		return Unet.model(x)
	end
end

"""
	function UnetBlock(in_ch::Int, inner_ch::Int; sub_level=x->x, layer_type="middle")

Create individual Unet layers which can be stacked.
	in_ch      - number of input channels
	kernel 	   - kernel size (kernel x kernel)
	inner_ch   - number of covnolutional masks/filters within this block/layer
	skip_ch    - number of convolutional masks/filters in skip connection
	sub_level  - lower Unet block or some function like indentity (x->x)
	layer_type - type of Unet block ["bottom", "middle", "top"]
"""
function UnetBlock(in_ch::Int, kernel::Int, inner_ch::Int, skip_ch::Int; sub_level=x->x, layer_type="middle")
	if layer_type == "top"
		pass = false
		down = Chain(Conv((kernel, kernel), in_ch=>inner_ch; stride=2, pad=1), BatchNorm(inner_ch))
		up = Chain(x->relu.(x), ConvTranspose((kernel, kernel), skip_ch+inner_ch=>in_ch; stride=2, pad=1))

		return UnetBlock(Chain(down, sub_level, up), pass, nothing)

	elseif layer_type == "bottom"
		pass = true
        down = Chain(x->leakyrelu.(x, 0.2f0),
                Conv((kernel, kernel), in_ch=>inner_ch; stride=2, pad=1),
                BatchNorm(inner_ch)
            )
        up = Chain(x->relu.(x),
                ConvTranspose((kernel, kernel), inner_ch=>in_ch; stride=2, pad=1),
                BatchNorm(in_ch)
            )

        skip_pass = Flux.Chain(
            Conv((3, 3), in_ch=>skip_ch, stride=1, pad=1),
            BatchNorm(skip_ch),
            x->leakyrelu.(x)
        )
        return UnetBlock(Chain(down, sub_level, up), pass, skip_pass)
        
	elseif layer_type == "middle"
		pass = true
        down = Chain(x->leakyrelu.(x, 0.2f0),
                Conv((kernel, kernel), in_ch=>inner_ch; stride=2, pad=1),
                BatchNorm(inner_ch)
            )
        up = Chain(x->relu.(x),
                ConvTranspose((kernel, kernel), skip_ch+inner_ch=>in_ch; stride=2, pad=1),
                BatchNorm(in_ch)
            )

        skip_pass = Flux.Chain(
            Conv((3, 3), inner_ch=>skip_ch, stride=1, pad=1),
            BatchNorm(skip_ch),
            x->leakyrelu.(x)
        )
        return UnetBlock(Chain(down, sub_level, up), pass, skip_pass)
    else
		error("unkown layer type")
	end
end



"""
	Unet
"""

struct Unet
	model
end

Flux.@functor Unet

function (Unet::Unet)(x)
	return Unet.model(x)
end

"""
	function Unet(isize::Int, in_ch::Int, nf::Int)

Create the convolutional generator (Encoder-Decoder) with skip connections with Unet shape
	isize      - size of output image (must be divisible by 32 i.e isize%32==0)
	in_ch      - number of input channels
	nf         - number of covnolutional masks/filters
	sf		   - number of convolutional masks in skip connections

Example: isize=32, in_ch=3, nf=64

	3=>64                        ->                     64+64=>3
		|                                                  ^
		v                                                  |
		64=>128                  ->               128+128=>64
			 |                                         ^
			 v                                         |
			 128=>256            ->          256+256=>128
				   |                              ^
				   v                              |
				  256=>512       ->     512+512=>256
						|                 ^
						v                 |
						512=>512 -> 512=>512
"""
function Unet(kernel::Int, in_ch::Int, nf::Int, sf::Int)
	unet = UnetBlock(nf, kernel, nf, sf, layer_type="bottom")
	unet = UnetBlock(nf, kernel, nf, sf, sub_level=unet, layer_type="middle")
	unet = UnetBlock(nf, kernel, nf, sf, sub_level=unet, layer_type="middle")
	unet = UnetBlock(nf, kernel, nf, sf, sub_level=unet, layer_type="middle")
	unet = UnetBlock(in_ch, kernel, nf, sf, sub_level=unet, layer_type="top")
	return Unet(unet)
end

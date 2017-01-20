function out_img = blend_mask(img, mask, color_val, alpha)
	r = size(img,1);
	c = size(img,2);

	img = im2double(img);
	
	mask_columns = zeros(r*c,3);
	fg_pix = find(mask == 1);
	mask_columns(fg_pix,:) = repmat(color_val, length(fg_pix),1);
            
	img = reshape(reshape(img,r*c,[]),[],3);    
	out_img = img;
	out_img(fg_pix,:) = alpha*out_img(fg_pix,:) + (1-alpha)*mask_columns(fg_pix,:);
	out_img = reshape(out_img,r,c,3);  
end

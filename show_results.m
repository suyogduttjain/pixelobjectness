data_dir = './images/';
input_file = './image_list.txt';
output_file = './output_list.txt';

image_files    = textread(input_file,'%s');
image_prefixes = textread(output_file,'%s');
num_images = length(image_files);

for i = 1:num_images
	feature_name = [image_prefixes{i} '_blob_0.mat'];
	data = load(fullfile(data_dir, feature_name));
	raw_result = data.data;
	
	img = imread(fullfile(data_dir,image_files{i}));
	img_row = min(size(img, 1),size(raw_result,1));
	img_col = min(size(img, 2),size(raw_result,2));
	raw_result = permute(raw_result, [2 1 3]);
	
	probs = raw_result(1:img_row, 1:img_col, :);
	[~, mask] = max(probs,[],3);
	mask = logical(mask-1);
	
	viz_img = blend_mask(img,mask,[0 1 0],0.5);
	
	figure(1);
	imshow(viz_img);
	title('Foreground Segmentation');
	
	figure(2);
	imagesc(probs(:,:,2));
	title('Pixel Objectness');
	pause;

end

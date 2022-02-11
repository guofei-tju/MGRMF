function save_results(filename,results)
fid = fopen(filename,'wt');
[m,n] = size(results);
if isa(results,'double')
    for i = 1:1:m
        for j = 1:1:n
            if j==n
                fprintf(fid,'%f\n',results(i,j));
            else
                fprintf(fid,'%f,',results(i,j));
            end
        end
    end
elseif isa(results,'cell')
    for i = 1:1:m
        for j = 1:1:n
            if j==n
                fprintf(fid,'%f\n',results{i,j});
            else
                if isa(results{i,j},'char')
                    fprintf(fid,'%s,',results{i,j});
                else
                    fprintf(fid,'%f,',results{i,j});
                end
            end
        end
    end
end
fclose(fid);
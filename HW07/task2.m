function task2()
    % task2 – Plot timing data for two block‐size configurations and save as PDF
    %
    % Expects two files in the working directory:
    %   timing2_1024.txt   % first row header, then columns [i time_ms]
    %   timing2_512.txt    % first row header, then columns [i time_ms]
    %
    % Produces task2.pdf with both curves on the same axes.

    % 1) Load the data, skipping the header line
    data1024 = dlmread('timing2_1024.txt','',1,0);
    data512  = dlmread('timing2_512.txt','',1,0);

    % Columns: [i time_ms]
    x1024 = data1024(:,1);
    t1024 = data1024(:,2);
    x512  = data512(:,1);
    t512  = data512(:,2);

    % 2) Plot both data sets on one figure
    figure('Position',[300 300 700 500]);
    hold on;
    plot(x1024, t1024, '-o', 'LineWidth',1.5, 'MarkerSize',6, ...
         'DisplayName','1024 threads/block');
    plot(x512,  t512,  '-s', 'LineWidth',1.5, 'MarkerSize',6, ...
         'DisplayName','512 threads/block');
    hold off;
    grid on;

    % 3) Labels, title, legend
    xlabel('i (block size exponent, threads per block = 2^i)');
    ylabel('Time (ms)');
    title('Task 2: Kernel Timing vs. Block‐Size Exponent');
    legend('Location','best');

    % 4) Save as PDF
    set(gcf,'PaperPositionMode','auto');
    print(gcf,'task2','-dpdf');
end

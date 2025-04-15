function task2()
    % -------------------------------------------------------------
    % 1) Load data from two text files (timing_16.txt, timing_512.txt)
    %    If you have a header row, skip it by specifying '1,0' in dlmread
    % -------------------------------------------------------------
    data1024  = dlmread('timing2_1024.txt','',1,0);   % skip 1 row, 0 cols
    data512 = dlmread('timing2_512.txt','',1,0);  % skip 1 row, 0 cols

    % Each data file has columns: [i time_ms first_val last_val]
    i1024    = data1024(:,1);
    time1024  = data1024(:,2);

    i512    = data512(:,1);
    time512 = data512(:,2);

    % -------------------------------------------------------------
    % 2) Create a figure and plot both data sets
    % -------------------------------------------------------------
    figure('Position',[300 300 700 500]);  % optional size
    hold on;  % so we can plot both lines

    plot(i1024,  time1024,  '-o', 'LineWidth',1.5,...
         'MarkerSize',6, 'DisplayName','1024 Threads/block');
    plot(i512, time512, '-s', 'LineWidth',1.5,...
         'MarkerSize',6, 'DisplayName','512 Threads/block');

    hold off;
    grid on;

    % -------------------------------------------------------------
    % 3) Label the axes, set a title, and place the legend
    % -------------------------------------------------------------
    xlabel('Threads per block = 2^x');
    ylabel('Time (ms)');
    title('Task 2 Timing Results');
    legend('Location','best');

    % -------------------------------------------------------------
    % 4) Save the figure as PDF
    % -------------------------------------------------------------
    print('-dpdf','task2.pdf');
end
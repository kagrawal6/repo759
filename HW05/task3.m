function task3()
    % -------------------------------------------------------------
    % 1) Load data from two text files (timing_16.txt, timing_512.txt)
    %    If you have a header row, skip it by specifying '1,0' in dlmread
    % -------------------------------------------------------------
    data16  = dlmread('timing_16.txt','',1,0);   % skip 1 row, 0 cols
    data512 = dlmread('timing_512.txt','',1,0);  % skip 1 row, 0 cols

    % Each data file has columns: [i time_ms first_val last_val]
    i16     = data16(:,1);
    time16  = data16(:,2);

    i512    = data512(:,1);
    time512 = data512(:,2);

    % -------------------------------------------------------------
    % 2) Create a figure and plot both data sets
    % -------------------------------------------------------------
    figure('Position',[300 300 700 500]);  % optional size
    hold on;  % so we can plot both lines

    plot(i16,  time16,  '-o', 'LineWidth',1.5,...
         'MarkerSize',6, 'DisplayName','16 Threads');
    plot(i512, time512, '-s', 'LineWidth',1.5,...
         'MarkerSize',6, 'DisplayName','512 Threads');

    hold off;
    grid on;

    % -------------------------------------------------------------
    % 3) Label the axes, set a title, and place the legend
    % -------------------------------------------------------------
    xlabel('Exponent i (Array Size = 2^i)');
    ylabel('Time (ms)');
    title('Task 3 Timing Results');
    legend('Location','best');

    % -------------------------------------------------------------
    % 4) Save the figure as PDF
    % -------------------------------------------------------------
    print('-dpdf','task3.pdf');
end

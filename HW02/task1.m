function task1()
    % task1.m
    %
    % Reads "timing_data.txt" (with a header) and plots time (ms)
    % vs. log2(n). The x-axis will be the exponent of 2 (10..30),
    % and the y-axis will be the measured time in milliseconds.

    % (1) Read the data, skipping the header row.
    M = dlmread('timing_data.txt', '', 1, 0);
    % Columns: [n, time_ms, first_val, last_val]
    n       = M(:,1);
    time_ms = M(:,2);

    % (2) Convert n to log2(n)
    x = log2(n);   % e.g., if n=1024, x=10

    % (3) Plot in a new figure: x vs. time_ms (linear scales)
    figure;
    plot(x, time_ms, 'o-','LineWidth',1.5);
    grid on;

    % (4) Set integer tick marks from 10..30 along the x-axis.
    %     Adjust this if your data extends beyond that range.
    xticks(10:1:30);

    % Label the axes
    xlabel('n (2^x)');  % or just "x = log2(n)"
    ylabel('Time (ms)');
    title('Scan Scaling Analysis');

    % (5) Save the figure as PDF
    print('-dpdf','task1.pdf');
    disp('Created task1.pdf');
end

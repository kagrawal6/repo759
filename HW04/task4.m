function task4()
    % -------------------------------------------------------------
    % 1) Load data from three text files (static, dynamic, guided)
    %    Skip the header row by specifying '1, 0' in dlmread
    % -------------------------------------------------------------
    staticData  = dlmread('timing_problem4_static.txt','',1,0);
    dynamicData = dlmread('timing_problem4_dynamic.txt','',1,0);
    guidedData  = dlmread('timing_problem4_guided.txt','',1,0);

    % Each data file has two columns: [threads, time_ms]
    threads_static  = staticData(:,1);
    time_static     = staticData(:,2);

    threads_dynamic = dynamicData(:,1);
    time_dynamic    = dynamicData(:,2);

    threads_guided  = guidedData(:,1);
    time_guided     = guidedData(:,2);

    % -------------------------------------------------------------
    % 2) Create a figure and plot all three on the same axes
    % -------------------------------------------------------------
    figure('Position',[300 300 700 500]);  % optional size
    hold on;  % so we can plot multiple lines on the same axes

    plot(threads_static,  time_static,  '-o', 'LineWidth',1.5,...
        'MarkerSize',6, 'DisplayName','Static Scheduling');
    plot(threads_dynamic, time_dynamic, '-s', 'LineWidth',1.5,...
        'MarkerSize',6, 'DisplayName','Dynamic Scheduling');
    plot(threads_guided,  time_guided,  '-^', 'LineWidth',1.5,...
        'MarkerSize',6, 'DisplayName','Guided Scheduling');

    hold off;
    grid on;

    % -------------------------------------------------------------
    % 3) Label the axes, set a title, and show the legend
    % -------------------------------------------------------------
    xlabel('Number of Threads');
    ylabel('Time (ms)');

    title('N-body Simulation (N = 800,Time = 100)');
    legend('Location','best');  % choose an optimal place for the legend

    % -------------------------------------------------------------
    % 4) (Optional) Save as PDF or PNG
    % -------------------------------------------------------------
    print('-dpdf', 'task4.pdf');
    % or: print('-dpng','scheduling_policies.png');
end

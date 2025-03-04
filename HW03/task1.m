function task1()
    % Read the data, skipping the header
    M = dlmread('timing_data.txt', '', 1, 0);
    % Columns: [t, time_ms, first_val, last_val]
    t = M(:,1);
    time_ms = M(:,2);

    % Plot time vs. t
    figure;
    plot(t, time_ms, 'o-','LineWidth',1.5);
    grid on;
    
    % Labels
    xlabel('Threads (t)');
    ylabel('Time (ms)');
    title('Task1: Execution Time vs Threads for mmul');

    % Save figure as PDF
    print('-dpdf', 'task1.pdf');
    disp('Created task1.pdf');
end

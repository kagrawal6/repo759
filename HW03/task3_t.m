function task3_t()
    % Read the data, skipping the header
    M = dlmread('timing_t.txt', '', 1, 0);
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
    title('Task3: Execution Time vs Threads for msort when ts = 256');

    % Save figure as PDF
    print('-dpdf', 'task3_t.pdf');
    disp('Created task3_t.pdf');
end

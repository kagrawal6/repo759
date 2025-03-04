function task3_ts()
    % Read the data, skipping the header
    M = dlmread('timing_ts.txt', '', 1, 0);
    % Columns: [t, ts, time_ms, first_val, last_val]
    ts = M(:,2);
    time_ms = M(:,3);

    % Plot time vs. ts (log scale for ts)
    figure;
    semilogx(ts, time_ms, 'o-','LineWidth',1.5);
    grid on;

    % Labels
    xlabel('Threshold (ts) (log scale)');
    ylabel('Time (ms)');
    title('Task3: Exec time vs Threshold for msort when n = 10^6 and t = 8');

    % Save figure as PDF
    print('-dpdf', 'task3_ts.pdf');
    disp('Created task3_ts.pdf');
end

%% ========================================================================
%  SILAHGRID 8-POD INTELLIGENT PLASMA COOLING NETWORK
%  Complete thermal management system with diverse scenarios
%% ========================================================================

clear all; close all; clc;

%% SYSTEM CONFIGURATION
disp('╔════════════════════════════════════════════════════════╗');
disp('║  SilahGrid Multi-Pod Plasma Cooling System                                                 ║');
disp('╚════════════════════════════════════════════════════════╝');

num_pods = 8;
pod_length = 6; pod_width = 2.4; pod_height = 2.6;
grid_res = 15;

% Temperature thresholds
T_ambient = 45;      % Desert ambient
T_plasma_on = 55;    % Activate plasma
T_plasma_off = 45;   % Deactivate plasma
T_critical = 75;     % Critical temperature

num_steps = 70; dt = 4.2;

%% PHYSICAL CONSTANTS
k_air = 0.028; rho_air = 1.2; cp_air = 1005;
alpha = k_air/(rho_air*cp_air);
dx = pod_length/(grid_res-1);

%% POD GRID SETUP
[X, Y, Z] = meshgrid(linspace(0, pod_length, grid_res), ...
                     linspace(0, pod_width, grid_res), ...
                     linspace(0, pod_height, grid_res));

%% PROCESSOR ZONES
processor_zones = false(size(X));
processor_zones(6:10, 8:12, 10:15) = true;   % Main bank
processor_zones(12:15, 8:12, 8:12) = true;   % Secondary

%% PLASMA ELECTRODE CONFIGURATION
num_electrodes = 8;
electrode_x = linspace(0.8, pod_length-0.8, num_electrodes);
electrode_y = pod_width/2 * ones(1,num_electrodes);
electrode_z = pod_height*0.9 * ones(1,num_electrodes);
E_field = 6000;

disp('>>> Computing EHD airflow patterns...');
u_base = zeros(size(X)); v_base = zeros(size(X)); w_base = zeros(size(X));

for i = 1:num_electrodes
    dist = sqrt((X-electrode_x(i)).^2 + (Y-electrode_y(i)).^2 + (Z-electrode_z(i)).^2);
    E_local = E_field ./ (dist.^2 + 0.05);
    
    u_base = u_base + 1.0*E_local.*(X-electrode_x(i))./(dist+0.05);
    v_base = v_base + 1.0*E_local.*(Y-electrode_y(i))./(dist+0.05);
    w_base = w_base - 4.0*E_local;
end

velocity_magnitude = sqrt(u_base.^2 + v_base.^2 + w_base.^2);
max_velocity = 5.5;
u_base = u_base./(velocity_magnitude + 0.01) * max_velocity;
v_base = v_base./(velocity_magnitude + 0.01) * max_velocity;
w_base = w_base./(velocity_magnitude + 0.01) * max_velocity;

%% DIVERSE POD SCENARIOS
disp('>>> Initializing 8 pods with diverse scenarios...');

pod_scenarios = {
    'Light Load - Steady', ...           % Pod 1: Low stable load
    'Heavy AI Training', ...             % Pod 2: High continuous load
    'Burst Processing', ...              % Pod 3: Intermittent spikes
    'Medium Continuous', ...             % Pod 4: Moderate steady
    'Idle/Standby', ...                  % Pod 5: Very low load
    'Critical Overload', ...             % Pod 6: Near critical
    'Variable Analytics', ...            % Pod 7: Fluctuating
    'Peak Performance' ...               % Pod 8: Maximum load
};

% Initial temperatures (diverse starting points)
max_temps_initial = [52, 88, 68, 78, 48, 90, 65, 92];

% Workload characteristics
workload_base = [0.6, 1.3, 0.9, 1.0, 0.4, 1.35, 0.85, 1.4];
workload_amplitude = [0.1, 0.2, 0.5, 0.15, 0.05, 0.25, 0.4, 0.3];
workload_frequency = [0.03, 0.05, 0.12, 0.04, 0.02, 0.06, 0.08, 0.07];
workload_phases = linspace(0, 2*pi, num_pods);

%% INITIALIZE POD STATES
T_pods = repmat(T_ambient, [grid_res, grid_res, grid_res, num_pods]);

for i = 1:num_pods
    Tp = T_pods(:,:,:,i);
    Tp(processor_zones) = max_temps_initial(i);
    % Add spatial temperature gradient
    for z = 1:grid_res
        heat_factor = (z/grid_res)^2;
        Tp(:,:,z) = Tp(:,:,z) + heat_factor * (max_temps_initial(i) - T_ambient) * 0.3;
    end
    T_pods(:,:,:,i) = Tp;
    
    fprintf('   Pod %d: %s (Initial: %.1f°C)\n', i, pod_scenarios{i}, max_temps_initial(i));
end

plasma_active = false(1, num_pods);
plasma_power = zeros(1, num_pods);
activation_count = zeros(1, num_pods);
total_cooling_time = zeros(1, num_pods);

%% DATA STORAGE
max_temps = zeros(num_pods, num_steps);
avg_temps = zeros(num_pods, num_steps);
plasma_status = zeros(num_pods, num_steps);
power_consumption = zeros(num_pods, num_steps);
cooling_efficiency = zeros(num_pods, num_steps);
time_vec = (0:num_steps-1) * dt;

%% MAIN SIMULATION LOOP
disp('>>> Starting multi-pod simulation...');

fig = figure('Position', [50, 50, 1800, 950], 'Color', 'w');

for step = 1:num_steps
    t = time_vec(step);
    
    for pod = 1:num_pods
        T = T_pods(:,:,:,pod);
        
        %% WORKLOAD MODELING (Diverse patterns)
        workload_variation = workload_base(pod) + ...
                           workload_amplitude(pod) * sin(t * workload_frequency(pod) + workload_phases(pod));
        
        % Add random spikes for burst scenarios
        if pod == 3 || pod == 7  % Burst/Variable pods
            if rand() < 0.05
                workload_variation = workload_variation + 0.4 * rand();
            end
        end
        
        % Critical pod stays hot
        if pod == 6 || pod == 8
            workload_variation = max(workload_variation, 1.2);
        end
        
        workload_variation = max(0.3, min(1.5, workload_variation));
        
        Q_gen = zeros(size(T));
        Q_gen(processor_zones) = 750 * workload_variation;
        
        %% TEMPERATURE METRICS
        current_max_temp = max(T(:));
        current_avg_temp = mean(T(:));
        
        %% INTELLIGENT PLASMA CONTROL
        prev_status = plasma_active(pod);
        
        if current_max_temp >= T_critical
            % CRITICAL MODE
            plasma_active(pod) = true;
            plasma_power(pod) = 1.0;  % Full power
            pod_status = 'CRITICAL';
            pod_color = [1 0 0];
            
        elseif plasma_active(pod)
            % ACTIVE COOLING MODE
            if current_avg_temp <= T_plasma_off
                plasma_active(pod) = false;
                plasma_power(pod) = max(0, plasma_power(pod) - 0.15);
                pod_status = 'COOLING DOWN';
                pod_color = [0 0.7 0];
            else
                plasma_power(pod) = min(1.0, plasma_power(pod) + 0.12);
                pod_status = 'ACTIVE';
                pod_color = [0 1 1];
            end
            
        else
            % MONITORING MODE
            if current_max_temp >= T_plasma_on
                plasma_active(pod) = true;
                plasma_power(pod) = 0.1;
                pod_status = 'ACTIVATING';
                pod_color = [1 0.8 0];
            else
                plasma_power(pod) = max(0, plasma_power(pod) - 0.08);
                pod_status = 'STANDBY';
                pod_color = [0.5 0.5 0.5];
            end
        end
        
        % Count activations
        if plasma_active(pod) && ~prev_status
            activation_count(pod) = activation_count(pod) + 1;
        end
        
        if plasma_power(pod) > 0.5
            total_cooling_time(pod) = total_cooling_time(pod) + dt;
        end
        
        %% ADAPTIVE COOLING STRENGTH
        temp_excess = max(0, current_max_temp - T_plasma_off);
        temp_range = max_temps_initial(pod) - T_plasma_off;
        cooling_factor = plasma_power(pod) * min(1, temp_excess / temp_range);
        
        u = u_base * cooling_factor;
        v = v_base * cooling_factor;
        w = w_base * cooling_factor;
        velocity_mag = sqrt(u.^2 + v.^2 + w.^2);
        
        %% HEAT TRANSFER PHYSICS
        [dTdx, dTdy, dTdz] = gradient(T, dx);
        laplacian_T = del2(T, dx);
        advection = u.*dTdx + v.*dTdy + w.*dTdz;
        
        plasma_cooling_boost = 3.8;
        cooling_rate = velocity_mag * 1.3 * plasma_cooling_boost;
        natural_cooling = 0.06 * (T - T_ambient);
        
        dT = dt * (alpha * laplacian_T * 1.6 ...
                   - advection ...
                   + Q_gen/(rho_air*cp_air) ...
                   - cooling_rate ...
                   - natural_cooling);
        
        T = T + dT;
        T(T < T_ambient) = T_ambient;
        T(T > 105) = 105;
        
        T_pods(:,:,:,pod) = T;
        
        %% RECORD METRICS
        max_temps(pod, step) = current_max_temp;
        avg_temps(pod, step) = current_avg_temp;
        plasma_status(pod, step) = plasma_power(pod);
        power_consumption(pod, step) = plasma_power(pod) * 5000;  % Watts
        
        % Cooling efficiency
        if max_temps_initial(pod) > T_ambient
            cooling_efficiency(pod, step) = (max_temps_initial(pod) - current_avg_temp) / ...
                                           (max_temps_initial(pod) - T_ambient) * 100;
        end
        
        pod_info{pod} = struct('status', pod_status, 'color', pod_color, ...
                              'max_temp', current_max_temp, ...
                              'avg_temp', current_avg_temp, ...
                              'plasma_pct', plasma_power(pod)*100);
    end
    
    %% VISUALIZATION (Every 7 steps)
    if mod(step, 7) == 0 || step == 1
        clf;
        
        % Define consistent colors for all subplots
        colors_plot = lines(num_pods);
        
        % 8 POD TEMPERATURE MAPS (More aesthetic styling)
        for pod = 1:num_pods
            subplot(3, 4, pod);
            mid_slice = squeeze(T_pods(:, :, round(grid_res/2), pod));
            
            % Create aesthetic temperature visualization
            h = imagesc(linspace(0, pod_length, grid_res), ...
                   linspace(0, pod_width, grid_res), mid_slice);
            colormap(gca, jet); 
            caxis([T_ambient, 100]);
            
            % Smooth interpolation for better aesthetics
            set(h, 'AlphaData', 0.9);
            
            % Enhanced status indicator with glow effect
            hold on;
            if strcmp(pod_info{pod}.status, 'CRITICAL')
                % Red critical border with glow
                rectangle('Position',[0 0 pod_length pod_width], ...
                         'EdgeColor',[1 0 0],'LineWidth',5,'LineStyle','-');
                rectangle('Position',[-0.05 -0.05 pod_length+0.1 pod_width+0.1], ...
                         'EdgeColor',[1 0.3 0.3],'LineWidth',2,'LineStyle','-');
            elseif plasma_power(pod) > 0.5
                % Cyan active border with glow
                rectangle('Position',[0 0 pod_length pod_width], ...
                         'EdgeColor',[0 1 1],'LineWidth',4);
                rectangle('Position',[-0.05 -0.05 pod_length+0.1 pod_width+0.1], ...
                         'EdgeColor',[0.5 1 1],'LineWidth',1.5,'LineStyle','-');
            elseif plasma_power(pod) > 0.1
                % Yellow ramping border
                rectangle('Position',[0 0 pod_length pod_width], ...
                         'EdgeColor',[1 1 0],'LineWidth',3);
            end
            hold off;
            
            % Enhanced title with better formatting
            title_str = sprintf('POD %d: %.1f°C\n%s | Plasma: %.0f%%\n%s', ...
                pod, pod_info{pod}.max_temp, pod_info{pod}.status, ...
                pod_info{pod}.plasma_pct, pod_scenarios{pod});
            
            title(title_str, 'FontWeight','bold', 'Color', pod_info{pod}.color, ...
                  'FontSize', 8, 'BackgroundColor', [0.95 0.95 0.95], ...
                  'EdgeColor', pod_info{pod}.color, 'LineWidth', 1.5);
            
            xlabel('Length (m)', 'FontSize', 8, 'FontWeight', 'bold');
            ylabel('Width (m)', 'FontSize', 8, 'FontWeight', 'bold');
            axis image;
            
            % Enhanced mini colorbar
            c = colorbar('southoutside');
            c.FontSize = 7;
            c.FontWeight = 'bold';
            c.Label.String = 'Temperature (°C)';
            c.Label.FontWeight = 'bold';
            
            % Add temperature range text (moved higher to avoid overlap)
            text(0.5, -0.66, sprintf('Range: %.0f-%.0f°C', min(mid_slice(:)), max(mid_slice(:))), ...
                 'Units', 'normalized', 'FontSize', 7, 'Color', [0.3 0.3 0.3], ...
                 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
            
        end
        
        % PLASMA POWER TIMELINE (Enhanced aesthetics)
        subplot(3, 4, [9 10]);
        colors_plot = lines(num_pods);
        hold on;
        
        % Draw filled areas for better visibility
        for pod = 1:num_pods
            area(time_vec(1:step), plasma_status(pod, 1:step)*100, ...
                 'FaceColor', colors_plot(pod,:), 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        end
        
        % Draw lines on top
        for pod = 1:num_pods
            plot(time_vec(1:step), plasma_status(pod, 1:step)*100, ...
                 'Color', colors_plot(pod,:), 'LineWidth', 2.5);
        end
        
        yline(100, 'r--', 'Full Power', 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
        hold off;
        
        xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 11);
        ylabel('Plasma Power (%)', 'FontWeight', 'bold', 'FontSize', 11);
        title('Plasma Activation Timeline - All Pods', 'FontWeight', 'bold', 'FontSize', 12);
        ylim([0 110]);
        xlim([0, time_vec(end)]);
        grid on;
        box on;
        
        % Enhanced legend with colored boxes
        lgd = legend(arrayfun(@(x) sprintf('Pod %d', x), 1:num_pods, 'UniformOutput', false), ...
               'Location', 'eastoutside', 'FontSize', 8, 'FontWeight', 'bold');
        title(lgd, 'Pods', 'FontWeight', 'bold');
        
        % TEMPERATURE EVOLUTION (Enhanced)
        subplot(3, 4, 11);
        hold on;
        
        % Fill background zones
        fill([0 time_vec(end) time_vec(end) 0], [T_critical T_critical 105 105], ...
             [1 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
        fill([0 time_vec(end) time_vec(end) 0], [T_plasma_on T_plasma_on T_critical T_critical], ...
             [1 1 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.2);
        
        % Plot temperature lines
        for pod = 1:num_pods
            plot(time_vec(1:step), max_temps(pod, 1:step), ...
                 'Color', colors_plot(pod,:), 'LineWidth', 2.2);
        end
        
        % Threshold lines
        yline(T_critical, 'r-', 'Critical', 'LineWidth', 2.5, 'LabelHorizontalAlignment', 'left', ...
              'FontWeight', 'bold');
        yline(T_plasma_on, 'Color', [1 0.8 0], 'LineStyle', '--', 'LineWidth', 2, ...
              'Label', 'Activate', 'LabelHorizontalAlignment', 'left', 'FontWeight', 'bold');
        yline(T_plasma_off, 'b--', 'Deactivate', 'LineWidth', 2, ...
              'LabelHorizontalAlignment', 'left', 'FontWeight', 'bold');
        
        hold off;
        xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 11);
        ylabel('Max Temperature (°C)', 'FontWeight', 'bold', 'FontSize', 11);
        title('Temperature Trends - Control Zones', 'FontWeight', 'bold', 'FontSize', 12);
        ylim([T_ambient-5, 105]);
        xlim([0, time_vec(end)]);
        grid on;
        box on;
        
        % EFFICIENCY METRICS
        subplot(3, 4, 12);
        current_efficiency = cooling_efficiency(:, step);
        b = bar(1:num_pods, current_efficiency, 'FaceColor', 'flat', 'CData', colors_plot, ...
                'EdgeColor', 'k', 'LineWidth', 1.2);
        xlabel('Pod Number', 'FontWeight', 'bold', 'FontSize', 10);
        ylabel('Cooling Efficiency (%)', 'FontWeight', 'bold', 'FontSize', 10);
        title('Current Cooling Efficiency', 'FontWeight', 'bold', 'FontSize', 11);
        ylim([0 110]);
        grid on;
        xticks(1:num_pods);
        
        % Add efficiency values on bars (better positioning)
        for pod = 1:num_pods
            if current_efficiency(pod) > 5
                text(pod, current_efficiency(pod) + 6, sprintf('%.0f%%', current_efficiency(pod)), ...
                     'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
                     'FontSize', 7, 'Color', 'k');
            else
                text(pod, current_efficiency(pod) + 5, sprintf('%.0f%%', current_efficiency(pod)), ...
                     'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
                     'FontSize', 9, 'Color', 'k');
            end
        end
        
        sgtitle(sprintf('SilahGrid 8-Pod Network | Time: %.1fs | Step: %d/%d | Total Active: %d pods', ...
                t, step, num_steps, sum(plasma_power > 0.5)), ...
                'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.2 0 0.6]);
        
        drawnow;
        pause(0.0001);
    end
    
    % Progress indicator
    if mod(step, 25) == 0
        fprintf('   Progress: %.0f%% | Active pods: %d/%d\n', ...
                step/num_steps*100, sum(plasma_power > 0.5), num_pods);
    end
end

%% FINAL COMPREHENSIVE ANALYSIS
disp(' ');
disp('>>> Generating final analysis...');

fig2 = figure('Position', [100, 100, 1800, 1000], 'Color', 'w');

% TEMPERATURE HISTORY ALL PODS
subplot(3, 3, [1 2]);
colors = lines(num_pods);
hold on;
for pod = 1:num_pods
    plot(time_vec, max_temps(pod, :), 'Color', colors(pod,:), 'LineWidth', 2);
end
yline(T_critical, 'r--', 'Critical', 'LineWidth', 2.5,'FontWeight', 'bold');
yline(T_plasma_on, 'm--', 'Activate', 'LineWidth', 2.5,'FontWeight', 'bold');
yline(T_plasma_off, 'b--', 'Deactivate', 'LineWidth', 2.5,'FontWeight', 'bold');
hold off;
xlabel('Time (s)', 'FontWeight', 'bold', 'FontSize', 11);
ylabel('Temperature (°C)', 'FontWeight', 'bold', 'FontSize', 11);
title('Complete Temperature History - All Pods', 'FontWeight', 'bold', 'FontSize', 13);
legend([arrayfun(@(x) sprintf('Pod %d', x), 1:num_pods, 'UniformOutput', false), ...
        {'Critical', 'Activate', 'Deactivate'}], 'Location', 'eastoutside', 'FontSize', 8);
grid on;
ylim([T_ambient-5, 105]);

% PLASMA ACTIVATION PATTERNS
subplot(3, 3, 3);
imagesc(time_vec, 1:num_pods, plasma_status*100);
colormap(gca, 'hot');
colorbar;
xlabel('Time (s)', 'FontWeight', 'bold');
ylabel('Pod Number', 'FontWeight', 'bold');
title('Plasma Activation Heatmap', 'FontWeight', 'bold', 'FontSize', 12);
yticks(1:num_pods);
set(gca, 'YDir', 'normal');

% ENERGY CONSUMPTION
subplot(3, 3, 4);
total_energy = sum(power_consumption, 2) * dt / 3600;  % Wh
bar(1:num_pods, total_energy, 'FaceColor', [0.2 0.6 1], 'EdgeColor', 'k', 'LineWidth', 1.5);
xlabel('Pod Number', 'FontWeight', 'bold');
ylabel('Energy Consumed (Wh)', 'FontWeight', 'bold');
title('Total Energy Consumption per Pod', 'FontWeight', 'bold', 'FontSize', 12);
grid on;
xticks(1:num_pods);
for pod = 1:num_pods
    text(pod, total_energy(pod) +6 + max(total_energy)*0.03, ...
         sprintf('%.1f', total_energy(pod)), 'HorizontalAlignment', 'center', ...
         'FontWeight', 'bold', 'FontSize', 9);
end

% ACTIVATION STATISTICS
subplot(3, 3, 5);
bar(1:num_pods, activation_count, 'FaceColor', [1 0.6 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5);
xlabel('Pod Number', 'FontWeight', 'bold');
ylabel('Activation Count', 'FontWeight', 'bold');
title('Plasma Activation Cycles', 'FontWeight', 'bold', 'FontSize', 12);
grid on;
xticks(1:num_pods);
for pod = 1:num_pods
    text(pod, activation_count(pod) + 8, sprintf('%d', activation_count(pod)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
end

% COOLING TIME PERCENTAGE
subplot(3, 3, 6);
cooling_time_pct = (total_cooling_time / time_vec(end)) * 100;
bar(1:num_pods, cooling_time_pct, 'FaceColor', [0.4 0.8 0.4], 'EdgeColor', 'k', 'LineWidth', 1.5);
xlabel('Pod Number', 'FontWeight', 'bold');
ylabel('Active Time (%)', 'FontWeight', 'bold');
title('Plasma Active Time Percentage', 'FontWeight', 'bold', 'FontSize', 12);
grid on;
xticks(1:num_pods);
ylim([0 110]);
for pod = 1:num_pods
    text(pod, cooling_time_pct(pod) + 7, sprintf('%.1f%%', cooling_time_pct(pod)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 7);
end

% AVERAGE EFFICIENCY
subplot(3, 3, 7);
avg_efficiency = mean(cooling_efficiency, 2);
bar(1:num_pods, avg_efficiency, 'FaceColor', 'flat', 'CData', colors, 'EdgeColor', 'k', 'LineWidth', 1.5);
xlabel('Pod Number', 'FontWeight', 'bold');
ylabel('Average Efficiency (%)', 'FontWeight', 'bold');
title('Average Cooling Efficiency', 'FontWeight', 'bold', 'FontSize', 12);
grid on;
xticks(1:num_pods);
ylim([0 100]);

% FINAL TEMPERATURES
subplot(3, 3, 8);
final_temps = max_temps(:, end);
initial_temps = max_temps(:, 1);
b = bar(1:num_pods, [initial_temps, final_temps]);
b(1).FaceColor = [1 0.4 0.4];
b(2).FaceColor = [0.4 0.4 1];
b(1).EdgeColor = 'k';
b(2).EdgeColor = 'k';
b(1).LineWidth = 1.5;
b(2).LineWidth = 1.5;
xlabel('Pod Number', 'FontWeight', 'bold');
ylabel('Temperature (°C)', 'FontWeight', 'bold');
title('Initial vs Final Temperatures', 'FontWeight', 'bold', 'FontSize', 12);
legend('Initial', 'Final', 'Location', 'best');
grid on;
xticks(1:num_pods);

% PERFORMANCE SUMMARY
subplot(3, 3, 9);
axis off;
cla; % Clear the current axes
xlim([0 1]); % Set generic, clean limits
ylim([0 1]);
% Target inner width is 38 characters.
summary_text = sprintf([...
    '╔══════════════════════════════════════╗\n' ...
    '║   SYSTEM PERFORMANCE SUMMARY         ║\n' ...
    '╠══════════════════════════════════════╣\n']);

for pod = 1:num_pods
    temp_reduction = initial_temps(pod) - final_temps(pod);
    
    % Pod lines: Target length 38 chars
    % We need to align the number such that the '|' delimiter always falls in the same column.
    % Pod X: ΔT= (9 chars) + 7 (temp value, e.g., -30.1) + °C (2 chars) + | (1 char) + 1 (space)
    % Cycles field: 1 (space) + 2 (cycles) + cycles (7 chars)
    % Total length: 9 + 7 + 2 + 1 + 1 + 1 + 2 + 7 = 30 chars. Remaining padding: 8 chars.
    
    % Use right-justification for the temperature (7.1f) to align the decimal points
    % Use left-justification for the cycles (-2d)
    summary_text = [summary_text sprintf(...
        '║   Pod %d: ΔT=%7.1f°C | %-2d cycles    ║\n', ...
        pod, temp_reduction, activation_count(pod))];
end

total_system_energy = sum(total_energy);
avg_system_efficiency = mean(avg_efficiency);

% System Totals: Target length 38 chars
% We use right-justification for the number and careful padding for the rest.
summary_text = [summary_text sprintf([...
    '╠══════════════════════════════════════╣\n' ...
    '║ Total Energy: %7.1f Wh             ║\n' ... % 16 + 7 + 3 + 12 = 38
    '║ Avg Efficiency: %5.1f%%               ║\n' ... % 17 + 5 + 2 + 14 = 38
    '║ Simulation Time: %5.1fs              ║\n' ... % 19 + 5 + 1 + 13 = 38
    '║ Water Used: 0 L                      ║\n' ...
    '║ System Status: OPERATIONAL           ║\n' ...
    '╚══════════════════════════════════════╝'], ...
    total_system_energy, avg_system_efficiency, time_vec(end))];

text(0.5, 0.5, summary_text, 'FontSize', 9, 'FontName', 'Courier New', ...
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
     'FontWeight', 'bold', 'BackgroundColor', [0.95 0.95 1], ...
     'EdgeColor', 'k', 'LineWidth', 2);
     
% ---------------------------------------------
sgtitle('SilahGrid 8-Pod Network - Complete Performance Analysis', ...
        'FontSize', 16, 'FontWeight', 'bold');

%% SAVE RESULTS
save('silahgrid_8pod_results.mat', 'T_pods', 'max_temps', 'avg_temps', ...
     'plasma_status', 'power_consumption', 'cooling_efficiency', ...
     'time_vec', 'activation_count', 'total_cooling_time');

%% FINAL REPORT
disp(' ');
disp('╔════════════════════════════════════════════════════════╗');
disp('║            SIMULATION COMPLETE                                                             ║');
disp('╚════════════════════════════════════════════════════════╝');
fprintf('\nPOD-BY-POD SUMMARY:\n');
fprintf('─────────────────────────────────────────────────────\n');
for pod = 1:num_pods
    fprintf('Pod %d (%s):\n', pod, pod_scenarios{pod});
    fprintf('  Initial: %.1f°C → Final: %.1f°C (ΔT: %.1f°C)\n', ...
            initial_temps(pod), final_temps(pod), initial_temps(pod)-final_temps(pod));
    fprintf('  Activations: %d | Active Time: %.1f%% | Energy: %.2f Wh\n', ...
            activation_count(pod), cooling_time_pct(pod), total_energy(pod));
    fprintf('  Efficiency: %.1f%%\n\n', avg_efficiency(pod));
end
fprintf('─────────────────────────────────────────────────────\n');
fprintf('NETWORK TOTALS:\n');
fprintf('  Total Energy: %.2f Wh\n', total_system_energy);
fprintf('  Average Efficiency: %.1f%%\n', avg_system_efficiency);
fprintf('  Total Activations: %d\n', sum(activation_count));
fprintf('  Simulation Duration: %.1f seconds\n', time_vec(end));
disp('════════════════════════════════════════════════════════');

disp(' ');
disp('✓ Results saved to: silahgrid_8pod_results.mat');
disp('✓ Visualizations complete');
disp(' ');


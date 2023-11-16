clc;clear;close all;
load("neuron_cell.mat");
load("norm_directions.mat");
neuron_id = [93,2];
dn = sprintf('neruon(%d,%d)',neuron_id);
if ~exist(dn,'dir')
	mkdir(dn);
end
neuron_spikes = neuron_cell{neuron_id(1),neuron_id(2)};
for i = 1:length(direction_unique)
    spikes = neuron_spikes(i,:);
    figure('visible','off');
    subplot(2,1,1);
    plot_raster(spikes);
    title(sprintf('neruon(%d,%d), direction(%0.2f,%0.2f)',neuron_id,direction_unique(i,:)));
    subplot(2,1,2);
    plot_psth(spikes);
    hold off;
    saveas(gca,sprintf('./neruon(%d,%d)/direction(%0.2f,%0.2f).png',neuron_id,direction_unique(i,:)),'png');
%     close all;
end


function plot_raster(spikes)
% spikes: 指定方向指定神经元多个trial的spikes
% spikes: cell,1*trial*226
trial_num = sum(~cellfun(@isempty,spikes));
for i = 1:trial_num
    for j = 1:length(spikes{i})
        time_point = spikes{i}(j)*1000;
        plot([time_point time_point],[i-0.4 i+0.4], 'k',LineWidth=1.5);
        hold on;
    end

end

plot([0 0],[0 trial_num+1],'r');
xlim([-210,1010]);%ms
xticks(-200:100:1000);
ylim([0,trial_num+1]);
ylabel('trial');
end

% -200,1000
function plot_psth(spikes)
time_bin = 100;%ms
psth = zeros([1,1200/time_bin]);
trial_num = sum(~cellfun(@isempty,spikes));
time_range = -200:time_bin:900;
for i = 1:trial_num
    for j = 1:length(spikes{i})
        time_point = spikes{i}(j)*1000;
        for k = 1:length(time_range)
            bin_count = sum(time_point>=time_range(k)&time_point<time_range(k)+time_bin);
            psth(k) = psth(k)+bin_count;
        end
    end
end
for i = 1:length(psth)
    line([time_range(i) time_range(i)+100],[psth(i) psth(i)]);
    if i <length(psth)
        line([time_range(i)+100 time_range(i)+100],[psth(i) psth(i+1)]);
    end
    %     hold on;
end

xlim([-210,1010]);%ms
xticks(-200:100:1000);
ylim([0,max(psth)+10]);
xlabel('time(ms)');
ylabel('count');
end

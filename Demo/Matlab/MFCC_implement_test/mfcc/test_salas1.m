% EXAMPLE Simple demo of the MFCC function usage.
%
%   This script is a step by step walk-through of computation of the
%   mel frequency cepstral coefficients (MFCCs) from a speech signal
%   using the MFCC routine.
%
%   See also MFCC, COMPARE.

%   Author: Kamil Wojcicki, September 2011

%for loop for running through all sound files 3 THINGS TO CHANGE
clear all; close all; clc;
fnames = dir('../../Sound_test/Sounds/School_Fire_Alarm_Vars/*.wav'); %________CHANGE
numfids = length(fnames);

% Define variables
    Tw = 40;                % analysis frame duration (ms)
    Ts = 20;                % analysis frame shift (ms)
    alpha = 0.0;%0.97;           % preemphasis coefficient
    M = 20;                 % number of filterbank channels 
    C = 25;                 % number of cepstral coefficients
    L = 22;                 % cepstral sine lifter parameter
    LF = 300;               % lower frequency limit (Hz)
    HF = 2000;              % upper frequency limit (Hz)
    fpath = '../../Sound_test/Sounds/School_Fire_Alarm_Vars/'; %_____________CHANGE
    
for id = 1:numfids
%____EOFor loop setup
    % Clean-up MATLAB's environment
      

    
    
    wav_file = strcat(fpath,fnames(id).name);  % input audio filename


    % Read speech samples, sampling rate and precision from file
    [ speech, fs, nbits ] = wavread( wav_file );
    speech = speech(:,1); %grab first channel
    
    %segment wav file
    seg_length = 2.4; %in seconds
    samp_incr = seg_length * fs; 
    
    speech_total = speech;
    clear speech;
    speech = speech_total(1:samp_incr);
    init = 1;
    init_step = round(samp_incr / 2);
    samp_segs = round(size(speech_total,1) / init_step) + 2;
    tot_len = size(speech_total,1);
    for step = 1:samp_segs
        
        
        
        if step ~= 1 
            init = init + init_step;
            if init > tot_len
                break;
            end
            term_ind = init + samp_incr - 1;
            if term_ind > tot_len
                sp_temp = zeros(samp_incr,1);
                t_len = tot_len - init + 1;
                sp_temp(1:t_len) = speech_total(init:end);
                speech = sp_temp;
                
            else
                speech = speech_total(init:term_ind);
            end
        end
        


            % Feature extraction (feature vectors as columns)
            [ MFCCs, FBEs, frames ] = ...
                            mfcc( speech, fs, Tw, Ts, alpha, @hamming, [LF HF], M, C+1, L );

% DONT NEED PLOTS for this
        %{
            % Generate data needed for plotting 
            [ Nw, NF ] = size( frames );                % frame length and number of frames
            time_frames = [0:NF-1]*Ts*0.001+0.5*Nw/fs;  % time vector (s) for frames 
            time = [ 0:length(speech)-1 ]/fs;           % time vector (s) for signal samples 
            logFBEs = 20*log10( FBEs );                 % compute log FBEs for plotting
            logFBEs_floor = max(logFBEs(:))-50;         % get logFBE floor 50 dB below max
            logFBEs( logFBEs<logFBEs_floor ) = logFBEs_floor; % limit logFBE dynamic range

        
            % Generate plots
            figure('Position', [30 30 800 600], 'PaperPositionMode', 'auto', ... 
                      'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 

            subplot( 311 );
            plot( time, speech, 'k' );
            xlim( [ min(time_frames) max(time_frames) ] );
            xlabel( 'Time (s)' ); 
            ylabel( 'Amplitude' ); 
            title( 'Speech waveform'); 

            subplot( 312 );
            imagesc( time_frames, [1:M], logFBEs ); 
            axis( 'xy' );
            xlim( [ min(time_frames) max(time_frames) ] );
            xlabel( 'Time (s)' ); 
            ylabel( 'Channel index' ); 
            title( 'Log (mel) filterbank energies'); 

            subplot( 313 );
            imagesc( time_frames, [1:C], MFCCs(2:end,:) ); % HTK's TARGETKIND: MFCC
            %imagesc( time_frames, [1:C+1], MFCCs );       % HTK's TARGETKIND: MFCC_0
            axis( 'xy' );
            xlim( [ min(time_frames) max(time_frames) ] );
            xlabel( 'Time (s)' ); 
            ylabel( 'Cepstrum index' );
            title( 'Mel frequency cepstrum' );

            % Set color map to grayscale
            colormap( 1-colormap('gray') ); 

            % Print figure to pdf and png files
            print('-dpdf', sprintf('%s.pdf', mfilename)); 
            print('-dpng', sprintf('%s.png', mfilename)); 
        %}

        % EOF

        %{
        % generate fixed MFCCs
        init_mfsize = size(MFCCs, 2);
        if (init_mfsize > 800)
            MFCCs = MFCCs(:,1:1:800);

        else
            MFCC_temp = zeros(14,800);
            MFCC_temp = MFCCs;
            MFCCs = MFCC_temp;
        end
        %}

        %save MFCC matrix
        wav_file = fnames(id).name; %__________CHANGE
        name_size = size(wav_file,2);
        name_size = name_size - 4; %index filename minus file extension
        filename = wav_file(1:name_size);
        seg_num = sprintf('_%d',step);
        filename = strcat(filename,seg_num);
        mat_ex = '.mat';
        filename = strcat(filename,mat_ex);
        path = './MFCC_data/School_Fire_Alarm_Mat/';       %_________CHANGE
        path = strcat(path, filename);
        save(path, 'MFCCs')
    end

end
    
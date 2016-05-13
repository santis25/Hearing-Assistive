%[audio,fs,nbits] = mp3read('School_Fire_Alarm-Cullen_Card-202875844.mp3');
[y,Fs] = audioread('School_Fire_Alarm-Cullen_Card-202875844.mp3');
filename = 'School_Fire_Alarm.wav';
audiowrite(filename,y,Fs);

clear y  Fs;
[y,Fs] = audioread('School_Fire_Alarm.wav');
%sound(y,Fs);
t=[1/Fs:1/Fs:length(y)/Fs];
plot(t,y); %figure of School_Fire_Alarm.wav
title('School Fire Alarm.wav')

%8000 - 36380 first interval of alarm
figure
plot(t(8000:36380),y(8000:36380))
title('Schol Fire Alarm sample')

%Discrete Fast Fourier Transform on first sound of alarm
figure
suby = y(8000:36380);
plot(abs(fft([suby,Fs])))
title('DFT of School Fire Alarm sample')

%Fast Fourier Transform with frequency-axes respective to wavfile
figure
N = 28382;
f = Fs/N.*(0:N-1);
Y = fft(suby,N);
Y = abs(Y(1:N))./(N/2);
Y(Y==0) = eps;
Y = 20 * log10(Y);

plot (f,Y);
title('DFT with Frequency-axes for sample')
grid on;

%amplitude frequency
figure
L = N;
T = 1/Fs;
t = (0: L -1) * T;
Y = fft(suby,N);
f = Fs*(0:(L/2))/L;
P2 = abs(Y/L); % Two-sided spectrum
P1 = P2(1:L/2+1); %Single sided spectrum based on P2 and signal length
P1(2:end-1) = 2*P1(2:end-1);

plot(f,P1)
title('Single-Sided Amplitude Spectrum of School Fire Alarm Sample')
xlabel('f (Hz)')
ylabel('|P1(f)|')
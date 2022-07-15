classdef TDS_energySpread < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        Slice_energy_spread_tool        matlab.ui.Figure
        imageacquisitionorloadingfiledefindingtheroisanalysingPanel  matlab.ui.container.Panel
        ImagescreenshotButton           matlab.ui.control.Button
        DefinetheroiesontheimageLabel   matlab.ui.control.Label
        loaddatafileButton              matlab.ui.control.Button
        ORLabel                         matlab.ui.control.Label
        fileEditFieldLabel              matlab.ui.control.Label
        fileEditField                   matlab.ui.control.EditField
        firstimageCheckBox              matlab.ui.control.CheckBox
        secondimageCheckBox             matlab.ui.control.CheckBox
        sliceenergyspreadcalculationButton  matlab.ui.control.Button
        MBIanalysingButton              matlab.ui.control.Button
        Label                           matlab.ui.control.Label
        EnergycalibrationdispersionmeasurementPanel  matlab.ui.container.Panel
        energycollibrationButton        matlab.ui.control.Button
        dispersionEditFieldLabel        matlab.ui.control.Label
        dispersionEditField             matlab.ui.control.NumericEditField
        TextArea                        matlab.ui.control.TextArea
        L2sumVolEditFieldLabel          matlab.ui.control.Label
        L2sumVolEditField               matlab.ui.control.NumericEditField
        toEditFieldLabel                matlab.ui.control.Label
        toEditField                     matlab.ui.control.NumericEditField
        usetheDesigndispersionCheckBox  matlab.ui.control.CheckBox
        beamenergyEditFieldLabel        matlab.ui.control.Label
        beamenergyEditField             matlab.ui.control.NumericEditField
        readenergyinB2DCheckBox         matlab.ui.control.CheckBox
        mLabel                          matlab.ui.control.Label
        MeVLabel                        matlab.ui.control.Label
        Image                           matlab.ui.control.Image
    end

    
    methods (Access = private)
        
        function [sigma, mu] = gaussfit(app, x, y, sigma0, mu0 )
           
            
            
            % Maximum number of iterations
            Nmax = 50;
            
            if( length( x ) ~= length( y ))
                fprintf( 'x and y should be of equal length\n\r' );
                exit;
            end
            
            n = length( x );
            x = reshape( x, n, 1 );
            y = reshape( y, n, 1 );
            
            %sort according to x
            X = [x,y];
            X = sortrows( X );
            x = X(:,1);
            y = X(:,2);
            
            %Checking if the data is normalized
            dx = diff( x );
            dy = 0.5*(y(1:length(y)-1) + y(2:length(y)));
            s = sum( dx .* dy );
            if( s > 1.5 | s < 0.5 )
                %  fprintf( 'Data is not normalized! The pdf sums to: %f. Normalizing...\n\r', s );
                y = y ./ s;
            end
            
            X = zeros( n, 3 );
            X(:,1) = 1;
            X(:,2) = x;
            X(:,3) = (x.*x);
            
            
            % try to estimate mean mu from the location of the maximum
            [ymax,index]=max(y);
            mu = x(index);
            
            % estimate sigma
            sigma = 1/(sqrt(2*pi)*ymax);
            
            if( nargin == 3 )
                sigma = sigma0;
            end
            
            if( nargin == 4 )
                mu = mu0;
            end
            
            %xp = linspace( min(x), max(x) );
            
            % iterations
            for i=1:Nmax
                %    yp = 1/(sqrt(2*pi)*sigma) * exp( -(xp - mu).^2 / (2*sigma^2));
                %    plot( x, y, 'o', xp, yp, '-' );
                
                dfdsigma = -1/(sqrt(2*pi)*sigma^2)*exp(-((x-mu).^2) / (2*sigma^2));
                dfdsigma = dfdsigma + 1/(sqrt(2*pi)*sigma).*exp(-((x-mu).^2) / (2*sigma^2)).*((x-mu).^2/sigma^3);
                
                dfdmu = 1/(sqrt(2*pi)*sigma)*exp(-((x-mu).^2)/(2*sigma^2)).*(x-mu)/(sigma^2);
                
                F = [ dfdsigma dfdmu ];
                a0 = [sigma;mu];
                f0 = 1/(sqrt(2*pi)*sigma).*exp( -(x-mu).^2 /(2*sigma^2));
                a = (F'*F)^(-1)*F'*(y-f0) + a0;
                sigma = a(1);
                mu = a(2);
                
                if( sigma < 0 )
                    sigma = abs( sigma );
                    fprintf( 'Instability detected! Rerun with initial values sigma0 and mu0! \n\r' );
                    fprintf( 'Check if your data is properly scaled! p.d.f should approx. sum up to \n\r' );
                    % exit;
                end
            end
            
        end
            
        
end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            clc 
            clear all
            close all
             x=doocsread('XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.AMPLITUDE.SP.1');
             %x.data=660
              xmin=x.data-4.
              xmax=x.data+4.
             app.L2sumVolEditField.Value=xmin;
             app.toEditField.Value=xmax;


             
             
        end

        % Button pushed function: energycollibrationButton
        function energycollibrationButtonPushed(app, event)
            % Energy collibrations and measuring the dispersion
            %%%%%%%%%%%%%%%%%%%%%%
            % Collibration part:
            %%%%%%%%%%%%%%%%%%%%%'
            
           y0=doocsread('XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/B2/ENERGY.ALL');
            E0=y0.data;
            x=doocsread('XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.AMPLITUDE.SP.1');
             
%             
%             %%
              xmin=x.data-8.;
              xmax=x.data+8.;
% 
            
             app.L2sumVolEditField.Value=xmin;
             app.toEditField.Value=xmax;
            
            xmin=app.L2sumVolEditField.Value
            xmax=app.toEditField.Value
            i=0;
           x=xmin:2:xmax;
             for x1=xmin:2:xmax
                 i=i+1
                 wi=doocswrite('XFEL.RF/LLRF.SUMVOLTAGE_CTRL/L2/SUMVOLTAGE.AMPLITUDE.SP.1',x1);
                % y1=doocsread('XFEL.FEEDBACK/FT2.LONGITUDINAL/MONITOR13/MEAN');
                 %Y(i)=y1.data*0.0054;
                 pause(1)
                 for j=1:20
                    y=doocsread('XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/B2D/ENERGY.ALL');
                    y1 = doocsread('XFEL.DIAG/IMAGEANALYSIS/OTRA.473.B2D/ANALYSIS.Y.GAUSS_MEAN');
                    Eng(j)=y.data;
                    Y0(j)=y1.data*0.0054/1000;  %m 
                    pause(0.11)
                 end
              Engy(i)=mean(Eng);
              Y(i)=mean(Y0);  %m 
              stdE=std(Eng);
              stdy=std(Y0);
                 
             end
             
            
   p = polyfit(x,Y,1)
   f1=polyval(p,x);
    dispersion=(p(1)*E0);
            
             figure('Name','dispersion measurement')
             %plot(x,Y, 'o-'), hold on
             errorbar(x,Y, stdy/2, '--o'); hold on
             plot(x,f1, '--')
             xlabel('RF sum L2(v)')
             ylabel('y position on screen')
             title(['dispersion=', num2str(dispersion), ' m'] )
             hold off 
           
%      data saving 
             data_disp_me=struct
             data_disp_me.note='the measuremetn has included 20 shots';
             data_disp_me.energy=E0;
             data_disp_me.RF_sum_L2=x ;
             data_disp_me.y=Y ;
             data_disp_me.position_error=stdy;
             data_disp_me.dispersion= dispersion;
            app.dispersionEditField.Value=dispersion;
            assignin('base', 'dispersion', dispersion) ;
            assignin('base', 'data_disp_me', data_disp_me) ;
            
            % sending to logbook 

A=getframe(gcf);
imwrite(A.cdata, 'temporly.png');
result_log=hlc_send_to_logbook('title', 'dispersion in B2D', ...
     'image', 'temporly.png')

delete('temporly.png')

        end

        % Value changed function: usetheDesigndispersionCheckBox
        function usetheDesigndispersionCheckBoxValueChanged(app, event)
            value = app.usetheDesigndispersionCheckBox.Value;
             %global dispersion
            dispersion=0.68;
            app.dispersionEditField.Value =dispersion;
            
            dispersion=app.dispersionEditField.Value 
            assignin('base', 'dispersion', dispersion) ;
            
              E0=app.beamenergyEditField.Value;
            
            data_disp_me=struct
             data_disp_me.energy=E0;
             data_disp_me.RF_sum_L2=0 ;
             data_disp_me.y=0 ;
             data_disp_me.dispersion= dispersion;
            
            
            assignin('base', 'data_disp_me', data_disp_me) ;
        end

        % Button pushed function: ImagescreenshotButton
        function ImagescreenshotButtonPushed(app, event)
            dataCamera=doocsread('XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT');
            
            y0=doocsread('XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/B2/ENERGY.ALL');
            E0=y0.data
            image=dataCamera.data.val_val;
            
            figure(1)
                im = pcolor(image);
                set(im, 'EdgeColor', 'none');
              
%              roi = drawrectangle(gca);
%              l(1)=position(1);
%              l(2)=position(3);
             set(im, 'EdgeColor', 'none');
             roi=getrect;
             l(1)=roi(1);
             l(2)=roi(3)+roi(1);
             l(3)=roi(2);
             l(4)=roi(2)+roi(4);
            
             image = dataCamera.data.val_val;
             a=size(image);
             dispersion=app.dispersionEditField.Value;
              %xp= dataCamera.data.val_val.x;  % check this values
              %yp=dataCamera.data.val_val.y;      % check this values 
           xp=1:a(2);
           yp=1:a(1);
              Calibration_f=1
             x=Calibration_f*xp;
             y=(yp*abs(0.0052)/1000)/dispersion*E0;
            
            

            time=datestr(now,'yyyy_mm_dd_HH_MM')
            assignin('base', 'y', y)
            assignin('base', 'x', x)
            assignin('base', 'image', image)
            assignin('base', 'l', l)
            assignin('base', 'Calibration_f',  Calibration_f)
            assignin('base', 'time',  time)
            
            

        end

        % Button pushed function: loaddatafileButton
        function loaddatafileButtonPushed(app, event)
        %app.fileEditField.Value='~/matlab/data_TDS/TDS_m'
        cd ~
            address=app.fileEditField.Value
            
        %cd  /System/Volumes/Data/home/xfeloper/data/tds_longitudinal_profile/old
            cd (address) 
            filename=uigetfile;
            time=split(filename, '.');
            time=char(time(1)); 
            assignin('base', 'filename',  filename)
            assignin('base', 'time', time)
            
           
        end

        % Value changed function: firstimageCheckBox
        function firstimageCheckBoxValueChanged(app, event)
          
            value = app.firstimageCheckBox.Value;
           if app.firstimageCheckBox.Value
                filename = evalin('base','filename');
                load(filename);
                close
                image = datafile.out1st.image;
                image=medfilt2(image);
                %ax = axes(app.Slice_energy_spread_tool);
                figure(1)
                im = pcolor(image);
                set(im, 'EdgeColor', 'none');
                roi=getrect;
                l(1)=roi(1);
                l(2)=roi(3)+roi(1);
                l(3)=roi(2);
                l(4)=roi(2)+roi(4);
                
                E0=app.beamenergyEditField.Value;
                i=1;
                dispersion=app.dispersionEditField.Value;
                %dispersion=handles.dispersion
                xp= datafile.out1st.x;
                yp=datafile.out1st.y;
                Calibration_f=(-datafile.pixel)/(datafile.calib_1st.calib_poly(i)); %fs
                x=Calibration_f*xp;
                y=(yp*abs(datafile.pixel*1e-3))/dispersion*E0;
                assignin('base', 'y', y)
                assignin('base', 'x', x)
                assignin('base', 'image', image)
                assignin('base', 'l', l)
                assignin('base', 'Calibration_f',  Calibration_f)
           end
        end

        % Value changed function: secondimageCheckBox
        function secondimageCheckBoxValueChanged(app, event)
            value = app.secondimageCheckBox.Value;
            if app.secondimageCheckBox.Value
                filename = evalin('base','filename');
                load(filename);
                close
                image = datafile.out2nd.image;
                image=medfilt2(image);
                %ax = axes(app.Slice_energy_spread_tool);
                figure(1)
                im = pcolor(image);
                set(im, 'EdgeColor', 'none');
                
                roi=getrect;
                l(1)=roi(1);
                l(2)=roi(3)+roi(1);
                l(3)=roi(2);
                l(4)=roi(2)+roi(4);
                
                E0=app.beamenergyEditField.Value;
                i=1;
                dispersion=app.dispersionEditField.Value
                %dispersion=handles.dispersion
                xp= datafile.out2nd.x;
                yp=datafile.out2nd.y;
                Calibration_f=(-datafile.pixel)/(datafile.calib_2nd.calib_poly(i)); %fs
                x=Calibration_f*xp;
                y=(yp*abs(datafile.pixel*1e-3));%*E0/dispersion;
                assignin('base', 'y', y)
                assignin('base', 'x', x)
                assignin('base', 'image', image)
                assignin('base', 'l', l)
                assignin('base', 'Calibration_f',  Calibration_f)
            end
        end

        % Value changed function: beamenergyEditField, 
        % readenergyinB2DCheckBox
        function readenergyinB2DCheckBoxValueChanged(app, event)
            value = app.readenergyinB2DCheckBox.Value;
            E0=doocsread('XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/B2/ENERGY.ALL')
            energy0=E0.data;
            app.beamenergyEditField.Value =energy0
        end

        % Button pushed function: sliceenergyspreadcalculationButton
        function sliceenergyspreadcalculationButtonPushed(app, event)
            dispersion=app.dispersionEditField.Value;
            E0=app.beamenergyEditField.Value;
            
            %clear sigma f mu
            x = evalin('base','x');
            y = evalin('base','y');
            image = evalin('base','image');
            l = evalin('base','l');
            Calibration_f=evalin('base', 'Calibration_f');
            time=evalin('base', 'time');
            data_disp_me=evalin('base', 'data_disp_me');
            m=0;
            N=length(x);
            for n0=l(1):l(2)
                n=single(int16(n0));
                [sigma0, mu0] = gaussfit(app, y, double(image(:,n) ), 0, 0);
                TF=isnan(mu0);
                if TF==0
                    m=m+1;
                    sigma(m)=sigma0; % this is the size of the spot on the screan 
                    mu(m)= mu0*E0/dispersion;
                    
                    X(m)=x(n);
                end
            end
            %%%%%%%%%%
            % Errors: 
            
            emittance=0.8e-6; 
            beta_y=1.009;
            Betax_t=51.03;
            emitance_x=0.6e-6;
            V=61; phi=pi/2.33;
            %datafile.pixel*1e-3
            resoltion=1.1247e-05*E0/dispersion; 
            Error1=sqrt(0.51/E0*beta_y*emittance) *E0/dispersion;
            Error2=sqrt(dispersion^2*V^2*cos(phi)^2*0.51*Betax_t*emitance_x/E0^3)...
                *E0/dispersion;
            
            %%%%%%%%%%%
            sigma=sqrt((sigma*E0/dispersion).^2-Error1^2-Error2^2-resoltion^2);
            
            Y=y*E0/dispersion;
            figure('Name','enery spread and energy chirp analysis')
            subplot(2, 1 ,1);
            h=pcolor(x,Y,image(:, :));
            set(h, 'EdgeColor', 'none');
            hold on
            plot(X,mu,'--w','LineWidth',1.5);
            plot(X,sigma,'-r','LineWidth',1.5);
            xlabel('t (fs)');
            if (Calibration_f==1)
                xlabel('slice number');
            end
            ylabel('\Delta E (Mev)');
            hold off
            %%
            
            
            
            subplot(2, 1 ,2);
   
            plot(X,mu,'or','LineWidth',.5, 'DisplayName','\DeltaEnergy');
            hold on
            plot(X,sigma,'*b','LineWidth',1,'DisplayName','\sigma_E');
            ylabel('\Delta E (Mev)');
            set(gca,'ycolor','r') ;
            legend
            
            p = polyfit(X,mu,3);
            y1 = polyval(p,X);
            plot(X,y1, ':m','LineWidth',1.5, 'DisplayName','fiting');
            
            
            title(['\DeltaE=' num2str(p(1)),'t^3+' num2str(p(2)),'t^2+' num2str(p(3)),'t+' num2str(p(4))]);
            
            hold off
            
            RMS_Y=sqrt(mean((sigma).^2));
%             energy_spread=sigma;
%             Energy=mu;

           data_energySpread_Ans=struct; 
           data_energySpread_Ans.dispersion=dispersion;
           data_energySpread_Ans.image=image;
           data_energySpread_Ans.x=x
           data_energySpread_Ans.y=y;
           data_energySpread_Ans.sigma=sigma;
           data_energySpread_Ans.energyDev=mu;
           data_energySpread_Ans.rms_energy_spread_Mev=RMS_Y;
           data_energySpread_Ans.optical_error_MeV=Error1;
           data_energySpread_Ans.TDS_error_MeV=Error2;
           data_energySpread_Ans.mean_energy_spread_MeV= mean(sigma);
           data_energySpread_Ans.energy_curve_fit=p;


           
           filename=sprintf('ESM_%s.mat', time);
           cd  /System/Volumes/Data/home/xfeloper/data/tds_longitudinal_profile/energy_spread_measu/ 
           save(filename, 'data_energySpread_Ans', 'data_disp_me')

            
           % clc
            fprintf( 'mean energy spread(MeV)=%.4f\n', mean(sigma));
            fprintf( 'rms energy spread(MeV)=%.4f\n', RMS_Y);
            fprintf( 'optical error(MeV)=%.4f\n', Error1);
            fprintf( 'TDS error(MeV)=%.4f\n',Error2);
            
            
            text=['mean energy spread(MeV)=', num2str(mean(sigma)) '\r\n', ...
             'rms energy spread(MeV)=', num2str(RMS_Y), '\r\n', ...
             'data saved in:', filename]
     %   end
%%%%%%%%%
% sending to logbook 

A=getframe(gcf);
imwrite(A.cdata, 'temporly.png');
result_log=hlc_send_to_logbook('title', 'enery spread and energy chirp in B2D', ...
    'text', sprintf(text), 'image', 'temporly.png')

delete('temporly.png')
     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        end

        % Button pushed function: MBIanalysingButton
        function MBIanalysingButtonPushed(app, event)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %       TDS FFT 2D and 1D analyzing 
% %      N. S. Mirian 03/09/2020
% %           14/09/2020
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%clc
% 
%function FFT2d=twoDanalyzing(filename)
%filename = evalin('base','filename'); load(filename);
x = evalin('base','x');
y = evalin('base','y');
image = evalin('base','image');
Calibration_f=evalin('base', 'Calibration_f');
dispersion=app.dispersionEditField.Value;
l = evalin('base','l');
time=evalin('base', 'time');
i=1;% Q=250e-12
E0=app.beamenergyEditField.Value;%MeV

dt=Calibration_f;  %fs
TBP=0.411;           %  time-bandwidth product for Gausian fuanction 0.441
N=l(2)-l(1)+1
Nfft=N;
f=TBP/(dt) *1e3  ;  % THz
frquency=linspace(-1,1, Nfft)*f/2;   % THz    Range of frequency
datafilepixel=-0.0054
dE=abs(datafilepixel*1e-3)/dispersion*E0;
M=TBP/dE;
Nfft2=l(4)-l(3)+1
Energy_modulation=linspace(-1,1, Nfft2)*M /2;   % 1/MeV  range of the FFT y axis
% %%
figure('Name','MBI analysis')
subplot(1,3,1)
fig.Color='w'
set(gca,'Color','w')

h=pcolor(x(l(1):l(2)), y(l(3):l(4)), image(l(3):l(4),l(1):l(2)));
xlabel('t (fs)');
if (Calibration_f==1)
    xlabel('slice number');
end
ylabel('\Delta E (Mev)');
set(h, 'EdgeColor', 'none');
title('beam spot')
grid off
%%

m=0
for i=l(3):l(4)
    n=single(int16(i));
    m=m+1;
    fftimage(m,:)=abs(fftshift(fft(image(n,l(1):l(2)))));
end


subplot(1,3,2)

set(gca,'Color','w');

h=pcolor(frquency, y(l(3):l(4)), fftimage);
set(h, 'EdgeColor', 'none');
xlabel('Delta f(THz)');
if (Calibration_f==1)
    xlabel('Delta f(THz)* factor');
end
ylabel('\Delta E (Mev)');
title('1D FFT')
grid off
%%
FFT2d=abs(fftshift( fft2(image(l(3):l(4),l(1):l(2)))));

subplot(1,3,3)
set(gca,'Color','w');

h=pcolor(frquency,Energy_modulation,FFT2d);
set(h, 'EdgeColor', 'none');
xlabel('f(THz)');
if (Calibration_f==1)
    xlabel('Delta f(THz)* factor');
end
ylabel('m (Mev^{-1})');
title('2D FFT')
data_MBI_Ans=struct;
data_MBI_Ans.Energy_modulation_mod=Energy_modulation;
data_MBI_Ans.frquency=frquency;
data_MBI_Ans.l=l;
data_MBI_Ans.FFT2d=FFT2d;
data_MBI_Ans.FFT1d=fftimage;
data_MBI_Ans.image=image;




filename=sprintf('MBIA_%s.mat', time);
           cd  /System/Volumes/Data/home/xfeloper/data/tds_longitudinal_profile/energy_spread_measu/ 
           save(filename, 'data_MBI_Ans')
           
           
           
           
           %%%%%%%%%
% sending to logbook 
text=['data seved in :', filename]
A=getframe(gcf);
imwrite(A.cdata, 'temporly.png');
result_log=hlc_send_to_logbook('title', 'MBI in B2D', ...
    'text', sprintf(text), 'image', 'temporly.png')

delete('temporly.png')
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create Slice_energy_spread_tool and hide until all components are created
            app.Slice_energy_spread_tool = uifigure('Visible', 'off');
            app.Slice_energy_spread_tool.Position = [100 100 516 448];
            app.Slice_energy_spread_tool.Name = 'Slice energy spread tool ';

            % Create imageacquisitionorloadingfiledefindingtheroisanalysingPanel
            app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel = uipanel(app.Slice_energy_spread_tool);
            app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel.Title = '2- image acquisition or loading file, definding the rois, analysing ';
            app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel.BackgroundColor = [0.9412 0.9412 0.9412];
            app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel.FontWeight = 'bold';
            app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel.FontSize = 16;
            app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel.Position = [5 17 507 241];

            % Create ImagescreenshotButton
            app.ImagescreenshotButton = uibutton(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel, 'push');
            app.ImagescreenshotButton.ButtonPushedFcn = createCallbackFcn(app, @ImagescreenshotButtonPushed, true);
            app.ImagescreenshotButton.BackgroundColor = [0.9412 0.9412 0.302];
            app.ImagescreenshotButton.FontSize = 14;
            app.ImagescreenshotButton.FontWeight = 'bold';
            app.ImagescreenshotButton.FontAngle = 'italic';
            app.ImagescreenshotButton.Position = [303 144 196 24];
            app.ImagescreenshotButton.Text = 'Image screen shot ';

            % Create DefinetheroiesontheimageLabel
            app.DefinetheroiesontheimageLabel = uilabel(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel);
            app.DefinetheroiesontheimageLabel.FontSize = 14;
            app.DefinetheroiesontheimageLabel.Position = [1 186 215 22];
            app.DefinetheroiesontheimageLabel.Text = '* Define the roies on the image ';

            % Create loaddatafileButton
            app.loaddatafileButton = uibutton(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel, 'push');
            app.loaddatafileButton.ButtonPushedFcn = createCallbackFcn(app, @loaddatafileButtonPushed, true);
            app.loaddatafileButton.Position = [2 136 98 22];
            app.loaddatafileButton.Text = 'load data file ';

            % Create ORLabel
            app.ORLabel = uilabel(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel);
            app.ORLabel.FontSize = 16;
            app.ORLabel.FontWeight = 'bold';
            app.ORLabel.Position = [229 136 29 22];
            app.ORLabel.Text = 'OR';

            % Create fileEditFieldLabel
            app.fileEditFieldLabel = uilabel(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel);
            app.fileEditFieldLabel.HorizontalAlignment = 'right';
            app.fileEditFieldLabel.Position = [4 165 25 22];
            app.fileEditFieldLabel.Text = 'file';

            % Create fileEditField
            app.fileEditField = uieditfield(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel, 'text');
            app.fileEditField.Position = [44 165 208 22];
            app.fileEditField.Value = 'data/tds_longitudinal_profile';

            % Create firstimageCheckBox
            app.firstimageCheckBox = uicheckbox(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel);
            app.firstimageCheckBox.ValueChangedFcn = createCallbackFcn(app, @firstimageCheckBoxValueChanged, true);
            app.firstimageCheckBox.Text = 'first image';
            app.firstimageCheckBox.Position = [7 115 77 22];

            % Create secondimageCheckBox
            app.secondimageCheckBox = uicheckbox(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel);
            app.secondimageCheckBox.ValueChangedFcn = createCallbackFcn(app, @secondimageCheckBoxValueChanged, true);
            app.secondimageCheckBox.Text = 'second image';
            app.secondimageCheckBox.Position = [7 94 97 22];

            % Create sliceenergyspreadcalculationButton
            app.sliceenergyspreadcalculationButton = uibutton(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel, 'push');
            app.sliceenergyspreadcalculationButton.ButtonPushedFcn = createCallbackFcn(app, @sliceenergyspreadcalculationButtonPushed, true);
            app.sliceenergyspreadcalculationButton.BackgroundColor = [0.498 0.9412 0.7843];
            app.sliceenergyspreadcalculationButton.FontSize = 15;
            app.sliceenergyspreadcalculationButton.FontWeight = 'bold';
            app.sliceenergyspreadcalculationButton.Position = [15 54 243 29];
            app.sliceenergyspreadcalculationButton.Text = 'slice energy spread calculation ';

            % Create MBIanalysingButton
            app.MBIanalysingButton = uibutton(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel, 'push');
            app.MBIanalysingButton.ButtonPushedFcn = createCallbackFcn(app, @MBIanalysingButtonPushed, true);
            app.MBIanalysingButton.BackgroundColor = [0.0745 0.6235 1];
            app.MBIanalysingButton.FontSize = 15;
            app.MBIanalysingButton.FontWeight = 'bold';
            app.MBIanalysingButton.Position = [15 19 240 26];
            app.MBIanalysingButton.Text = 'MBI analysing';

            % Create Label
            app.Label = uilabel(app.imageacquisitionorloadingfiledefindingtheroisanalysingPanel);
            app.Label.Position = [122 100 25 22];
            app.Label.Text = '';

            % Create EnergycalibrationdispersionmeasurementPanel
            app.EnergycalibrationdispersionmeasurementPanel = uipanel(app.Slice_energy_spread_tool);
            app.EnergycalibrationdispersionmeasurementPanel.Title = '1- Energy calibration (dispersion measurement)';
            app.EnergycalibrationdispersionmeasurementPanel.BackgroundColor = [0.902 0.902 0.902];
            app.EnergycalibrationdispersionmeasurementPanel.FontWeight = 'bold';
            app.EnergycalibrationdispersionmeasurementPanel.FontSize = 16;
            app.EnergycalibrationdispersionmeasurementPanel.Position = [2 263 512 177];

            % Create energycollibrationButton
            app.energycollibrationButton = uibutton(app.EnergycalibrationdispersionmeasurementPanel, 'push');
            app.energycollibrationButton.ButtonPushedFcn = createCallbackFcn(app, @energycollibrationButtonPushed, true);
            app.energycollibrationButton.BackgroundColor = [0.8902 0.698 0.9294];
            app.energycollibrationButton.FontSize = 14;
            app.energycollibrationButton.FontWeight = 'bold';
            app.energycollibrationButton.Position = [282 119 150 24];
            app.energycollibrationButton.Text = 'energy collibration ';

            % Create dispersionEditFieldLabel
            app.dispersionEditFieldLabel = uilabel(app.EnergycalibrationdispersionmeasurementPanel);
            app.dispersionEditFieldLabel.HorizontalAlignment = 'right';
            app.dispersionEditFieldLabel.Position = [236 30 61 22];
            app.dispersionEditFieldLabel.Text = 'dispersion';

            % Create dispersionEditField
            app.dispersionEditField = uieditfield(app.EnergycalibrationdispersionmeasurementPanel, 'numeric');
            app.dispersionEditField.Position = [312 30 75 22];

            % Create TextArea
            app.TextArea = uitextarea(app.EnergycalibrationdispersionmeasurementPanel);
            app.TextArea.BackgroundColor = [0.9412 0.9412 0.9412];
            app.TextArea.Position = [10 61 237 82];
            app.TextArea.Value = {'* 1- Turn off the energy feedback on L2'; '* 2- turn off trajectory feebdack   '; '* 3- Turn on the camera 457 '; '* 4- Insert the screen '};

            % Create L2sumVolEditFieldLabel
            app.L2sumVolEditFieldLabel = uilabel(app.EnergycalibrationdispersionmeasurementPanel);
            app.L2sumVolEditFieldLabel.HorizontalAlignment = 'right';
            app.L2sumVolEditFieldLabel.Position = [254 87 65 22];
            app.L2sumVolEditFieldLabel.Text = 'L2 sum Vol';

            % Create L2sumVolEditField
            app.L2sumVolEditField = uieditfield(app.EnergycalibrationdispersionmeasurementPanel, 'numeric');
            app.L2sumVolEditField.Position = [334 87 62 22];

            % Create toEditFieldLabel
            app.toEditFieldLabel = uilabel(app.EnergycalibrationdispersionmeasurementPanel);
            app.toEditFieldLabel.HorizontalAlignment = 'right';
            app.toEditFieldLabel.Position = [405 87 25 22];
            app.toEditFieldLabel.Text = 'to';

            % Create toEditField
            app.toEditField = uieditfield(app.EnergycalibrationdispersionmeasurementPanel, 'numeric');
            app.toEditField.Position = [438 87 70 22];

            % Create usetheDesigndispersionCheckBox
            app.usetheDesigndispersionCheckBox = uicheckbox(app.EnergycalibrationdispersionmeasurementPanel);
            app.usetheDesigndispersionCheckBox.ValueChangedFcn = createCallbackFcn(app, @usetheDesigndispersionCheckBoxValueChanged, true);
            app.usetheDesigndispersionCheckBox.Text = 'use the Design dispersion';
            app.usetheDesigndispersionCheckBox.Position = [13 27 160 22];

            % Create beamenergyEditFieldLabel
            app.beamenergyEditFieldLabel = uilabel(app.EnergycalibrationdispersionmeasurementPanel);
            app.beamenergyEditFieldLabel.HorizontalAlignment = 'right';
            app.beamenergyEditFieldLabel.Position = [222 6 76 22];
            app.beamenergyEditFieldLabel.Text = 'beam energy';

            % Create beamenergyEditField
            app.beamenergyEditField = uieditfield(app.EnergycalibrationdispersionmeasurementPanel, 'numeric');
            app.beamenergyEditField.ValueChangedFcn = createCallbackFcn(app, @readenergyinB2DCheckBoxValueChanged, true);
            app.beamenergyEditField.Position = [313 6 74 22];
            app.beamenergyEditField.Value = 2403;

            % Create readenergyinB2DCheckBox
            app.readenergyinB2DCheckBox = uicheckbox(app.EnergycalibrationdispersionmeasurementPanel);
            app.readenergyinB2DCheckBox.ValueChangedFcn = createCallbackFcn(app, @readenergyinB2DCheckBoxValueChanged, true);
            app.readenergyinB2DCheckBox.Text = 'read energy in B2D';
            app.readenergyinB2DCheckBox.Position = [13 6 125 22];

            % Create mLabel
            app.mLabel = uilabel(app.EnergycalibrationdispersionmeasurementPanel);
            app.mLabel.Position = [410 30 25 22];
            app.mLabel.Text = 'm';

            % Create MeVLabel
            app.MeVLabel = uilabel(app.EnergycalibrationdispersionmeasurementPanel);
            app.MeVLabel.Position = [405 6 30 22];
            app.MeVLabel.Text = 'MeV';

            % Create Image
            app.Image = uiimage(app.Slice_energy_spread_tool);
            app.Image.Position = [377 385 248 55];
            app.Image.ImageSource = '2022-02-16T14 19 32-00.png';

            % Show the figure after all components are created
            app.Slice_energy_spread_tool.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = TDS_energySpread

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.Slice_energy_spread_tool)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.Slice_energy_spread_tool)
        end
    end
end
<h1>Baidu KDD CUP 2022</h1>
<h2>Spatial Dynamic Wind Power Forecasting Challenge</h2>
<p>Since 1997, KDD Cup has been the premier annual Data Mining competition held in conjunction with the ACM SIGKDD Conference on Knowledge Discovery and Data Mining. This year’s KDD Cup challenge task presents interesting technical challenges and has practical importance for the utilization of wind energy. Here we propose a spatial dynamic wind power forecasting challenge to facilitate the progress of data-driven machine learning methods for wind power forecasting.</p>
<h2>Motivation</h2>
<p>Wind Power Forecasting (WPF) aims to accurately estimate the wind power supply of a wind farm at different time scales. Wind power is a kind of clean and safe source of renewable energy, but cannot be produced consistently, leading to high variability. Such variability can present substantial challenges to incorporating wind power into a grid system. To maintain the balance between electricity generation and consumption, the fluctuation of wind power requires power substitution from other sources that might not be available at short notice (for example, usually it takes at least 6 hours to fire up a coal plant). Thus, WPF has been widely recognized as one of the most critical issues in wind power integration and operation. There has been an explosion of studies on wind power forecasting problems appearing in the data mining and machine learning community. Nevertheless, how to well handle the WPF problem is still challenging, since high prediction accuracy is always demanded to ensure grid stability and security of supply.</p>
<h2>Challenge Overview</h2>
<p><b>We present a unique Spatial Dynamic Wind Power Forecasting dataset from Longyuan Power Group Corp. Ltd: SDWPF, </b>which includes the spatial distribution of wind turbines, as well as the dynamic context factors like temporal, weather, and turbine internal status. Whereas, most of the existing datasets and competitions treat WPF as a time series prediction problem without knowing the locations and context information of wind turbines.</p>
<p>An illustration of the SDWPF dataset is shown below. Each wind turbine can generate the wind power Ti separately, and the outcome power of the wind farm is the sum of all the wind turbines. In other words, at time t, the output power of the wind farm is P=∑_i Patv_i .</p>
<b>There are two unique features for this competition task different from previous WPF competition settings:</b>
<ol>
<li>Spatial distribution: this competition will provide the relative location of all wind turbines given a wind farm for modeling the spatial correlation among wind turbines.</li>
<li>Dynamic context: important weather situations and turbine internal contexts monitored by each wind turbine are provided to facilitate the forecasting task.</li>
</ol>
<img src="http://bj.bcebos.com/v1/ai-studio-match/file/e33ed7955c9549ff9ef26c2b601b90f525b4f77c35ea44f29f9a58208b9a8cd9?authorization=bce-auth-v1%2F0ef6765c1e494918bc0d4c3ca3e5c6d1%2F2022-04-11T10%3A50%3A41Z%2F-1%2F%2Fae0788a9685f87715b2e2b0778189b21e98003f6aec74cf1e5b5eae06359dca5" alt="img" />
<h2>Schedule</h2>
<p>All the deadlines are at 23:59 AOE.</p>
<ul>
<li>March 16, Registration site open.</li>
<li>March 20, Initial data released. Participants will practice with the initial WPF data to get familiar with the problem.</li>
<li>April 10, Full data released. We will release all the datasets and baseline code.</li>
<li>May 10, Submission start. All teams can try the demonstration submission to ensure a smooth final test submission.</li>
<li>June 20, Test data update. A new test set will be released for the test prediction.</li>
<li>**June 21, Team Freeze Deadline. All team members should be confirmed. **</li>
<li>July 15, Final submission deadline. Each team submits its final prediction model. The models will be evaluated on a private test set to determine the candidate awardee teams.</li>
<li>July 18, Winner notification. Private notifications and instructions about the code& technical paper are sent to the awardees.</li>
<li>July 21, Code and the technical paper submission deadline for the awardees.</li>
<li>July 22, Winners Announcement.</li>
<li>August 1, Techincal paper revision deadline.</li>
<li>August 15, KDD Cup Workshop.</li>
</ul>
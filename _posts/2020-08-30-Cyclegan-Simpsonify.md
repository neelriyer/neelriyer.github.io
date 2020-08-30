# Can you turn yourself into a Simpsons Character?

<table><tr><td><img src='https://drive.google.com/uc?id=1pTuzTcVpPWZnvEtmd4FOg6vdtQ0zIWQh'></td><td><img src='https://drive.google.com/uc?id=1SV5vLt-KrXRmAesBagivP6OlzsFhbsyO'></td></tr></table>

<table><tr><td><img src='https://drive.google.com/uc?id=1jmpktn6Jj9ia_bKpkNzU3A-vtEjRrcpc'></td><td><img src='https://drive.google.com/uc?id=1_t_Gdq8FxdhkF7QVhP_RxCDvO9a-CLjB'></td></tr></table>


[Cyclegan](https://arxiv.org/abs/1703.10593) is a framework that is capable of image to image translatation. It's been applied in some really interesting cases. Such as converting [horses to zebras](https://camo.githubusercontent.com/69cbc0371777fba5d251a564e2f8a8f38d1bf43f/68747470733a2f2f6a756e79616e7a2e6769746875622e696f2f4379636c6547414e2f696d616765732f7465617365725f686967685f7265732e6a7067) (and back again) and photos of the winter to photos of the summer. 

I thought this could be potentially applied to The Simpsons. I was inspired by sites like https://turnedyellow.com/ and https://makemeyellow.photos/. 

So you would upload a photo of your face and Cyclegan would translate that into a Simpsons Character. 

In this article I describe the workflow required to  'Simpsonise' yourself using Cyclegan. It's worth noting that the [paper](https://arxiv.org/pdf/1703.10593.pdf) explicitly mentions that large geometric changes are usually unsuccessful. 

But I'm going to attempt this anyway.


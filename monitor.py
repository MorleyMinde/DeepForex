from btgym.monitor import BTgymMonitor

class Monitor:
    def __init__(self,dir):
        # Essense metrics for one step:
        self.step_monitor = BTgymMonitor(
            logdir='{}'.format(dir),
            subdir='/metrics/step'.format(dir),
                                    scalars={'step/reward',
                                             'step/drawdown',
                                             'step/broker_value',
                                            },
                                   )
        # ..for episode:
        self.episode_monitor = BTgymMonitor(
            logdir='{}'.format(dir),
            subdir='/episode',
                                       scalars={'process/episode/cpu_time_sec',
                                                'episode/value',
                                                'episode/reward'
                                               },
                                       images={'episode'},
                                      )
        # State pictures:
        self.images_monitor = BTgymMonitor(
            logdir='{}'.format(dir),
            subdir='/step/image',
            images={'human',},
                                     )

    def logstep(self,json):
        self.step_monitor.write(
                        feed_dict={'step/reward': json['reward'],
                                   'step/drawdown': json['drawdown'],
                                   'step/broker_value': json['broker_value'],
                                  },
                        global_step=json['steps'],
                        )
    def logepisode(self,json):
        self.episode_monitor.write(
                        feed_dict={
                            'process/episode/cpu_time_sec': json['cpu_time_sec'],
                            'episode/reward': json['reward'],
                            'episode/value': json['broker_value'],
                            'episode': json['episode'],
                                  },
                        global_step=json['global_step'],
                        )
    def logimage(self,feed_dict,global_step):
        self.images_monitor.write(
                feed_dict=feed_dict,
                global_step=global_step,
            )

    def close(self):
        self.step_monitor.close()
        self.episode_monitor.close()
        self.images_monitor.close()
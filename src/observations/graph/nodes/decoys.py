import CybORG.Shared.Enums as Enums

from .connection_node import ConnectionNode

def init_decoy(uuid, dtype):
    '''
    ConnectionNode factory. Creates the relevant kind of decoy 
    Args: 
        uuid: unique identifier (int)
        dtype: what kind of decoy we're making (str)
    '''
    d = {
        'apache2': dict(
            process_name=Enums.ProcessName.APACHE2,
            process_type=Enums.ProcessType.WEBSERVER
        ),
        'tomcat': dict(
            # For some reason, listed under proc version enum but not proc name
            process_name=Enums.ProcessName.UNKNOWN,
            process_type=Enums.ProcessType.WEBSERVER
        ),
        'vsftpd': dict(
            # Also not in the enum
            process_name=Enums.ProcessName.UNKNOWN, 
            process_type=Enums.ProcessType.WEBSERVER
        ),
        'haraka': dict(
            # Also in ProcessVersion but not ProcessName
            process_name=Enums.ProcessName.UNKNOWN,
            process_type=Enums.ProcessType.SMTP
        )
    }

    return ConnectionNode(uuid, d[dtype], is_decoy=True)

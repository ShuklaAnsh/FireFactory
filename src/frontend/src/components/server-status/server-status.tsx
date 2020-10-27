import React from 'react';
import "./server-status.scss";

export interface ServerStatus {
  ready: boolean;
  msg: string;
}

interface ServerStatusProps {
  serverStatus: ServerStatus
}

export const ServerStatus = (props: ServerStatusProps) => {
  return (
    <div id={"server_status_container"}>
      <div id={"server_status_text_container"}>
        <div id={"server_status_text"} className={props.serverStatus.ready ? "server_ready" : "server_not_ready"}>
          {props.serverStatus.msg}
        </div>
      </div>
    </div>
  )
}
{{/*
Expand the name of the chart.
*/}}
{{- define "hal-nemofinder.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully-qualified app name.
Truncated to 63 chars (DNS-1123 label limit).
*/}}
{{- define "hal-nemofinder.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "hal-nemofinder.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels applied to every object.
*/}}
{{- define "hal-nemofinder.labels" -}}
helm.sh/chart: {{ include "hal-nemofinder.chart" . }}
{{ include "hal-nemofinder.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: hal-nemofinder
{{- end -}}

{{/*
Selector labels — used to identify pods and services.
*/}}
{{- define "hal-nemofinder.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hal-nemofinder.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "hal-nemofinder.api.selectorLabels" -}}
{{ include "hal-nemofinder.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end -}}

{{- define "hal-nemofinder.worker.selectorLabels" -}}
{{ include "hal-nemofinder.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end -}}

{{/*
ServiceAccount name.
*/}}
{{- define "hal-nemofinder.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "hal-nemofinder.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Worker image coordinates — fall back to the main image if no override.
*/}}
{{- define "hal-nemofinder.workerImage" -}}
{{- $repo := default .Values.image.repository .Values.image.workerRepository -}}
{{- $tag  := default .Values.image.tag        .Values.image.workerTag -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}

{{- define "hal-nemofinder.apiImage" -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}

{{/*
Database URL helpers. When the embedded postgresql subchart is enabled,
point at the generated service; otherwise honour externalDatabase.*.
*/}}
{{- define "hal-nemofinder.databaseHost" -}}
{{- if .Values.postgresql.enabled -}}
{{ printf "%s-postgresql" .Release.Name }}
{{- else -}}
{{ .Values.externalDatabase.host }}
{{- end -}}
{{- end -}}

{{- define "hal-nemofinder.databaseUser" -}}
{{- if .Values.postgresql.enabled -}}
{{ .Values.postgresql.auth.username }}
{{- else -}}
{{ .Values.externalDatabase.user }}
{{- end -}}
{{- end -}}

{{- define "hal-nemofinder.databaseName" -}}
{{- if .Values.postgresql.enabled -}}
{{ .Values.postgresql.auth.database }}
{{- else -}}
{{ .Values.externalDatabase.database }}
{{- end -}}
{{- end -}}

{{/*
The name of the secret that holds the database password.
*/}}
{{- define "hal-nemofinder.databaseSecretName" -}}
{{- if .Values.postgresql.enabled -}}
{{- default (printf "%s-postgresql" .Release.Name) .Values.postgresql.auth.existingSecret -}}
{{- else -}}
{{ .Values.externalDatabase.existingSecret }}
{{- end -}}
{{- end -}}

{{- define "hal-nemofinder.databaseSecretKey" -}}
{{- if .Values.postgresql.enabled -}}
password
{{- else -}}
{{ .Values.externalDatabase.existingSecretPasswordKey }}
{{- end -}}
{{- end -}}

{{/*
Default anti-affinity spreading API pods across hosts for HA.
*/}}
{{- define "hal-nemofinder.api.defaultAffinity" -}}
podAntiAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        topologyKey: kubernetes.io/hostname
        labelSelector:
          matchLabels:
            {{- include "hal-nemofinder.api.selectorLabels" . | nindent 12 }}
    - weight: 50
      podAffinityTerm:
        topologyKey: topology.kubernetes.io/zone
        labelSelector:
          matchLabels:
            {{- include "hal-nemofinder.api.selectorLabels" . | nindent 12 }}
{{- end -}}

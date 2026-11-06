# Conditions

Conditions allow a template developer to create many scenarios using the same template. cfn-lint does not use parameter values to determine the validaty of a template so cfn-lint will try to validate all the possible scenarios that are created by using conditions.

# Using conditions

## Defining conditions

Conditions are defined in a section of your CloudFormation template. At the lowest level all conditions will equate to a set of `Fn::Equals` that compares two values. Advanced condition scenarios can be created using `Fn::And`, `Fn::Or`, `Fn::Not`. Conditions can be nested using `Fn::Condition` function.

```yaml
Parameters:
    Environment:
        Type: String
Conditions:
    IsProduction: !Equals [!Ref Environment, "prod"]
    IsDevelopment: !Equals [!Ref Environment, "dev"]
    IsProductionAndUsEast1:
        !And
        - !Condition IsProduction
        - !Equals [!Ref AWS::Region, "us-east-1"]
```

## Using conditions

Conditions can be used in two locations. 1/ Is the at the resource or output level defined by the attribute `Condition` 2/ Using the function `Fn::If` under a resources `Properties` property. `Fn::If` can be used at any layer as long as it is the only key in an object.

```yaml
Resources:
    HTTPSCertificate:
        Type: AWS::CertificateManager::Certificate
        Condition: HaveTargets
        Properties:
            DomainName: 'example.com'
            ValidationMethod: DNS

    CloudFrontAlias:
        Type: AWS::Route53::RecordSet
        Condition: CreateDNSRecords
        Properties:
            HostedZoneId: !Ref ServiceHostedZoneId
            Name: 'example.com'
            Type: A
            AliasTarget:
                DNSName: !GetAtt Distribution1.DomainName
                HostedZoneId: Z2FDTNDATAQYW2

    Distribution1:
        Type: AWS::CloudFront::Distribution
        Condition: HaveTargets
        Properties:
            DistributionConfig:
                Enabled: true
                ViewerCertificate:
                    AcmCertificateArn: !Ref HTTPSCertificate
                    SslSupportMethod: sni-only
                    MinimumProtocolVersion: TLSv1.2_2019
                DefaultCacheBehavior:
                    TargetOriginId: Service1
Outputs:
    DomainName:
        Value: !If [HaveTargets, !GetAtt Distribution1.DomainName, ""]
```


# Handling conditions

## Solving conditions
cfn-lint builds a boolean formula from the conditions and decides which combinations of condition values a template can take. For instance for conditions `IsProduction` and `IsDevelopment` on one parameter it will calculate the possibilities `True`/`False`, `False`/`True`, or `False`/`False`. `Fn::And` is converted to `And`, `Fn::Or` is converted to `Or`, and `Fn::Not` is converted to `Not`. Two `Fn::Equals` that compare the same parameter to different values can never be `True` together, and a parameter's `AllowedValues` and the template's `Rules` restrict the values further (a `Rules` assertion built from anything other than the condition functions, such as `Fn::Contains`, is not used). A question about the conditions is answered by searching the values the parameters it reads can take, or by a small SAT solver where that search would be too large.

